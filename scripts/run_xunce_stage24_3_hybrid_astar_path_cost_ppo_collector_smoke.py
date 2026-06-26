from __future__ import annotations

import argparse
import shutil
import json
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


CONFIG_SCHEMA_VERSION = "xunce-stage24-3-hybrid-astar-path-cost-ppo-collector-smoke-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage24-3-summary/v1"
AUDIT_SCHEMA_VERSION = "xunce-stage24-3-hybrid-path-transition-contract-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage24-3-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage24-3-manifest/v1"

STAGE_ID = "xunce-stage24-3-hybrid-astar-path-cost-ppo-collector-smoke"
DEFAULT_CONFIG = "configs/xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage24_hybrid_astar_pose_planner/"
    "outputs/path_feedback_batch_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke_v1"
)
DEFAULT_STAGE24_2_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage24_hybrid_astar_pose_planner/"
    "outputs/path_feedback_batch_xunce_stage24_2_hybrid_astar_reward_path_cost_contract_v1"
)

SUMMARY_FILE = "xunce-stage24-3-summary.json"
STAGE21_1_SUMMARY_FILE = "xunce-stage24-3-stage21-1-collector-summary.json"
STAGE21_2_SUMMARY_FILE = "xunce-stage24-3-stage21-2-reward-summary.json"
STAGE21_3_SUMMARY_FILE = "xunce-stage24-3-stage21-3-batch-summary.json"
AUDIT_FILE = "xunce-stage24-3-hybrid-path-transition-contract-audit.json"
ROUTING_FILE = "xunce-stage24-3-next-stage-routing.json"
REPORT_FILE = "xunce-stage24-3-report.md"
MANIFEST_FILE = "xunce-stage24-3-manifest.json"

SLOPE_COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
HYBRID_ASTAR_PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"
ROUTE_FROM_STAGE24_2 = "run_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke"
ROUTE_INPUTS = "rerun_stage24_3_required_inputs"
ROUTE_COLLECTOR = "repair_stage24_3_collector_hybrid_path_cost_contract"
ROUTE_BINDING = "repair_stage24_3_action_viewpoint_path_cost_binding"
ROUTE_REWARD = "repair_stage24_3_reward_path_cost_provenance"
ROUTE_BATCH = "repair_stage24_3_batch_gate"
ROUTE_STAGE24_4 = "run_stage24_4_hybrid_astar_path_cost_ppo_update_smoke"
ROUTE_BOUNDARY = "resolve_stage24_3_boundary_rejections"

BOUNDARY_FIELDS = (
    "stage24_3_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage24.3 Hybrid A* path-cost PPO collector smoke.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke(
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
    stage24_2_summary = _read_stage24_2_summary(config) if not input_reasons else {}

    stage21_1_summary: dict[str, Any] = {}
    stage21_2_summary: dict[str, Any] = {}
    stage21_3_summary: dict[str, Any] = {}
    if not boundary_reasons and not input_reasons:
        stage21_1_summary = _run_stage21_1(config, stage24_2_summary, output_root, repo_root)
        if stage21_1_summary.get("status") == "passed":
            stage21_2_summary = _run_stage21_2(config, output_root, repo_root)
        if stage21_2_summary.get("status") == "passed":
            stage21_3_summary = _run_stage21_3(config, output_root, repo_root)

    transitions = _read_jsonl_if_exists(output_root / "s21_1" / stage21_1.TRAINABLE_BATCH_FILE)
    rewards = _read_jsonl_if_exists(output_root / "s21_2" / stage21_2.EVALUATION_FILE)
    batch_rows = _read_jsonl_if_exists(output_root / "s21_3" / stage21_3.BATCH_FILE)
    rejections = _read_jsonl_if_exists(output_root / "s21_1" / stage21_1.REJECTION_FILE)
    stage21_1_config_used = _read_json_if_exists(output_root / "xunce-stage24-3-stage21-1-config.json")
    expected_source_roi_expansion_root = (
        stage21_1_config_used.get("source_roi_expansion_root")
        or (_high_res_root_from_stage24_2_summary(stage24_2_summary, repo_root) if stage24_2_summary else None)
    )
    stage21_1_manifest = _read_json_if_exists(output_root / "s21_1" / stage21_1.MANIFEST_FILE)
    audit = _transition_contract_audit(
        transitions,
        rewards,
        batch_rows,
        rejections,
        expected_source_roi_expansion_root=expected_source_roi_expansion_root,
        stage21_1_manifest=stage21_1_manifest,
    )
    status, route, route_reason = _route(boundary_reasons, input_reasons, stage21_1_summary, stage21_2_summary, stage21_3_summary, audit)
    blocking = _unique(boundary_reasons + input_reasons + _route_blocking_reasons(route, audit, stage21_1_summary, stage21_2_summary, stage21_3_summary))

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking,
        "stage24_2_root": config["stage24_2_root"],
        "stage24_2_status": stage24_2_summary.get("status"),
        "stage24_2_next_required_change": stage24_2_summary.get("next_required_change"),
        "stage21_1_root": str(output_root / "s21_1"),
        "stage21_1_status": stage21_1_summary.get("status"),
        "stage21_2_root": str(output_root / "s21_2"),
        "stage21_2_status": stage21_2_summary.get("status"),
        "stage21_3_root": str(output_root / "s21_3"),
        "stage21_3_status": stage21_3_summary.get("status"),
        "stage23_2b_high_res_root": audit["stage21_1_expected_source_roi_expansion_root"],
        "stage21_1_actual_source_roi_expansion_root": audit["stage21_1_actual_source_roi_expansion_root"],
        "stage21_1_source_roi_expansion_root_match": audit["stage21_1_source_roi_expansion_root_match"],
        "coverage_source": SLOPE_COVERAGE_SOURCE,
        "path_cost_source": HYBRID_ASTAR_PATH_COST_SOURCE,
        "platform_contract_id": stage24_2_summary.get("platform_contract_id") or config.get("platform_contract_id"),
        "platform_contract_hash": stage24_2_summary.get("platform_contract_hash") or config.get("platform_contract_hash"),
        "max_traversable_slope_deg": float(stage24_2_summary.get("max_traversable_slope_deg") or config["max_traversable_slope_deg"]),
        "required_scenario_count": int(config["required_scenario_count"]),
        "rollout_steps": int(config["rollout_steps"]),
        **_summary_counts(audit),
        "stage24_3_authorized": False,
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
        "stage24_3_authorized": False,
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
        "hybrid_path_transition_contract_audit": str(output_root / AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / STAGE21_1_SUMMARY_FILE, stage21_1_summary)
    _write_json(output_root / STAGE21_2_SUMMARY_FILE, stage21_2_summary)
    _write_json(output_root / STAGE21_3_SUMMARY_FILE, stage21_3_summary)
    _write_json(output_root / AUDIT_FILE, audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _run_stage21_1(config: dict[str, Any], stage24_2_summary: dict[str, Any], output_root: Path, repo_root: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage21_1_base_config"], repo_root)
    high_res_root = _high_res_root_from_stage24_2_summary(stage24_2_summary, repo_root)
    safe_high_res_root = _prepare_safe_high_res_source_root(Path(high_res_root), output_root) if high_res_root else None
    cfg.update(
        {
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "dynamic_validation_work_root": str(output_root / "_dynamic_validation_work_stage24_3"),
            "theta_aware_candidate_viewpoints_enabled": True,
            "theta_bin_count": int(config["theta_bin_count"]),
            "theta_step_deg": int(config["theta_step_deg"]),
            "sensor_model_id": config["sensor_model_id"],
            "sensor_fov_deg": float(config["sensor_fov_deg"]),
            "sensor_range_cells": int(config["sensor_range_cells"]),
            "slope_obstacle_aware_theta_reward_enabled": True,
            "hybrid_astar_pose_path_cost_enabled": True,
            "hybrid_astar_goal_position_tolerance_m": float(config.get("hybrid_astar_goal_position_tolerance_m", 2.0)),
            "hybrid_astar_primitive_duration_s": float(config.get("hybrid_astar_primitive_duration_s", 1.5)),
            "hybrid_astar_max_speed_mps": float(config.get("hybrid_astar_max_speed_mps", 3.0)),
            "obstacle_occlusion_enabled": True,
            "derive_slope_blocked_cells_from_sidecar_dem": True,
            "max_traversable_slope_deg": float(stage24_2_summary.get("max_traversable_slope_deg") or config["max_traversable_slope_deg"]),
            "platform_contract_id": stage24_2_summary.get("platform_contract_id") or config.get("platform_contract_id"),
            "platform_contract_hash": stage24_2_summary.get("platform_contract_hash") or config.get("platform_contract_hash"),
            "platform_max_climb_deg": float(stage24_2_summary.get("max_traversable_slope_deg") or config["max_traversable_slope_deg"]),
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
    if safe_high_res_root:
        cfg["source_roi_expansion_root"] = str(safe_high_res_root)
    cfg_path = output_root / "xunce-stage24-3-stage21-1-config.json"
    _write_json(cfg_path, cfg)
    return stage21_1.run_xunce_stage21_1_on_policy_ppo_rollout_collector(
        config_path=cfg_path,
        output_root=output_root / "s21_1",
        repo_root=repo_root,
    )


def _high_res_root_from_stage24_2_summary(summary: dict[str, Any], repo_root: Path) -> str | None:
    explicit = summary.get("high_res_root") or summary.get("source_roi_expansion_root")
    if isinstance(explicit, str) and explicit.strip() and Path(explicit).is_dir():
        return str(Path(explicit))
    stage23_2b_root = summary.get("stage23_2b_root")
    if isinstance(stage23_2b_root, str) and stage23_2b_root.strip():
        high_res = _high_res_root_from_stage23_2b_root(Path(stage23_2b_root))
        if high_res:
            return high_res
    # Prefer the repaired Stage23.5A high-res root when it exists: it has the
    # two-scenario coverage Stage24.3 needs. Stage24.3 still repairs invalid
    # start cells locally before invoking the collector.
    stage23_5a_high_res = Path(
        "D:/CodexDownloads/lunar-path-planning/stage23_slope_obstacle_aware_theta_reward/"
        "outputs/path_feedback_batch_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage_v1/"
        "s23_2b/a30/hr"
    )
    if stage23_5a_high_res.is_dir():
        return str(stage23_5a_high_res)
    # Real Stage24.2 was replay-only and may not carry Stage23 lineage yet.
    # Fall back to the current platform-aligned Stage23.2B root so Stage24.3
    # can still prove that the collector writes lineage directly.
    default_stage23_2b = Path(
        "D:/CodexDownloads/lunar-path-planning/stage23_platform_alignment/"
        "outputs/path_feedback_batch_xunce_stage23_2b_platform_geometry_sensor_contract_alignment_v1"
    )
    if default_stage23_2b.is_dir():
        return _high_res_root_from_stage23_2b_root(default_stage23_2b)
    return None


def _prepare_safe_high_res_source_root(source_root: Path, output_root: Path) -> Path:
    source_root = source_root.resolve()
    safe_root = output_root / "stage24_3_safe" / "high_res_roi_expansion"
    safe_sidecar_dir = safe_root / "path_planner_sidecars"
    safe_root.mkdir(parents=True, exist_ok=True)
    safe_sidecar_dir.mkdir(parents=True, exist_ok=True)

    row_updates: dict[str, dict[str, Any]] = {}
    slices_path = source_root / "xunce-high-fidelity-real-map-slices.jsonl"
    if slices_path.is_file():
        updated_rows: list[dict[str, Any]] = []
        for row in _read_jsonl_if_exists(slices_path):
            if not isinstance(row, dict):
                continue
            updated = dict(row)
            sidecar_src = Path(str(row.get("sidecar", "")))
            sidecar_data = _read_json_if_exists(sidecar_src)
            safe_start = _nearest_safe_start_cell(sidecar_data, row.get("start_cell") or row.get("current_cell") or [0, 0])
            original_start = _cell_list(row.get("start_cell") or row.get("current_cell") or [0, 0])
            if safe_start is not None:
                updated["start_cell"] = safe_start
                updated["current_cell"] = safe_start
                updated["stage24_3_safe_start_cell_repair"] = {
                    "original_start_cell": original_start,
                    "repaired_start_cell": safe_start,
                    "reason": "original_start_cell_blocked_by_30deg_slope_obstacle",
                }
                seed = updated.get("scenario_seed", "seed")
                updated["scenario_variant_id"] = f"{updated.get('scenario_id')}-seed-{seed}-start-{safe_start[0]}-{safe_start[1]}"
            if sidecar_src.is_file():
                sidecar_dst = safe_sidecar_dir / sidecar_src.name
                shutil.copy2(sidecar_src, sidecar_dst)
                updated["sidecar"] = str(sidecar_dst)
            contract_src = Path(str(row.get("contract", "")))
            if contract_src.is_file():
                contract_dst = safe_sidecar_dir / contract_src.name
                shutil.copy2(contract_src, contract_dst)
                updated["contract"] = str(contract_dst)
            scenario_id = str(updated.get("scenario_id", ""))
            if scenario_id:
                row_updates[scenario_id] = updated
            updated_rows.append(updated)
        _write_jsonl(safe_root / slices_path.name, updated_rows)

    for path in source_root.iterdir():
        if path.is_dir():
            continue
        if path.name == "xunce-high-fidelity-real-map-slices.jsonl":
            continue
        if path.name == "xunce-high-fidelity-path-feedback-audit.json":
            audit = _read_json_if_exists(path)
            if isinstance(audit.get("scenarios"), list):
                scenarios = []
                for scenario in audit["scenarios"]:
                    if not isinstance(scenario, dict):
                        continue
                    updated = dict(scenario)
                    row_update = row_updates.get(str(updated.get("scenario_id")))
                    if row_update:
                        for field in (
                            "start_cell",
                            "current_cell",
                            "scenario_variant_id",
                            "sidecar",
                            "contract",
                            "stage24_3_safe_start_cell_repair",
                        ):
                            if field in row_update:
                                updated[field] = row_update[field]
                    scenarios.append(updated)
                audit["scenarios"] = scenarios
                audit["scenario_count"] = len(scenarios)
            _write_json(safe_root / path.name, audit)
        else:
            shutil.copy2(path, safe_root / path.name)
    return safe_root


def _nearest_safe_start_cell(sidecar: dict[str, Any], original_start: Any) -> list[int] | None:
    if not isinstance(sidecar, dict):
        return _cell_list(original_start)
    cost = sidecar.get("cost")
    if not isinstance(cost, list) or not cost or not isinstance(cost[0], list):
        return _cell_list(original_start)
    height = len(cost)
    width = len(cost[0])
    passable_mask = sidecar.get("passable_mask")
    if not isinstance(passable_mask, list) or len(passable_mask) != height:
        passable_mask = [[True for _ in range(width)] for _ in range(height)]
    blocked = {tuple(cell) for cell in sidecar.get("slope_blocked_cells", []) or []}
    blocked.update(tuple(cell) for cell in sidecar.get("blocked_cells", []) or [])
    blocked.update(tuple(cell) for cell in sidecar.get("obstacle_cells", []) or [])
    original = _cell_list(original_start) or [0, 0]
    safe_cells: set[tuple[int, int]] = set()
    for y in range(height):
        row = passable_mask[y] if isinstance(passable_mask[y], list) else []
        for x in range(width):
            if x >= len(row) or not bool(row[x]) or (x, y) in blocked:
                continue
            safe_cells.add((x, y))
    if not safe_cells:
        return original
    if (int(original[0]), int(original[1])) in safe_cells:
        return original
    component = _largest_safe_component(safe_cells)
    if not component:
        return original
    centroid_x = sum(cell[0] for cell in component) / float(len(component))
    centroid_y = sum(cell[1] for cell in component) / float(len(component))
    x, y = min(
        component,
        key=lambda cell: (
            abs(float(cell[0]) - centroid_x) + abs(float(cell[1]) - centroid_y),
            cell[1],
            cell[0],
        ),
    )
    return [int(x), int(y)]


def _largest_safe_component(safe_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    remaining = set(safe_cells)
    largest: list[tuple[int, int]] = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            x, y = stack.pop()
            for dx, dy in neighbors:
                neighbor = (x + dx, y + dy)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        if len(component) > len(largest):
            largest = component
    return largest


def _cell_list(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError):
        return None


def _high_res_root_from_stage23_2b_root(stage23_2b_root: Path) -> str | None:
    stage23_2a_summary_path = stage23_2b_root / "xunce-stage23-2b-rerun-stage23-2a-summary.json"
    if not stage23_2a_summary_path.is_file():
        return None
    stage23_2a_summary = _read_json(stage23_2a_summary_path)
    high_res_root = stage23_2a_summary.get("high_res_root")
    if not isinstance(high_res_root, str) or not high_res_root.strip():
        return None
    high_res_path = Path(high_res_root)
    return str(high_res_path) if high_res_path.is_dir() else None


def _run_stage21_2(config: dict[str, Any], output_root: Path, repo_root: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage21_2_base_config"], repo_root)
    cfg.update(
        {
            "stage21_1_collector_root": str(output_root / "s21_1"),
            "coverage_first_reward_profile": config["coverage_first_reward_profile"],
            "require_theta_aware_reward_contract": True,
            "slope_obstacle_aware_theta_reward_enabled": True,
            "require_hybrid_astar_path_cost_contract": True,
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
    cfg_path = output_root / "xunce-stage24-3-stage21-2-config.json"
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
            "stage21_3_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    cfg_path = output_root / "xunce-stage24-3-stage21-3-config.json"
    _write_json(cfg_path, cfg)
    return stage21_3.run_xunce_stage21_3_ppo_batch_validation(
        config_path=cfg_path,
        output_root=output_root / "s21_3",
        repo_root=repo_root,
    )


def _transition_contract_audit(
    transitions: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
    batch_rows: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    *,
    expected_source_roi_expansion_root: str | None,
    stage21_1_manifest: dict[str, Any],
) -> dict[str, Any]:
    transition_missing = 0
    mask_mismatch = 0
    sampling_contract_mismatch = 0
    binding_mismatch = 0
    hybrid_binding_mismatch = 0
    strict_false = 0
    hybrid_transition_missing = 0
    default_astar_replaced_count = 0
    ackermann_claimed_count = 0
    hard_risk = 0
    mask_violation = 0
    open_grid = 0
    for row in transitions:
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        viewpoints = info.get("candidate_viewpoints")
        action_index = _int_or_none(row.get("action_index"))
        slope_present = _slope_arrays_present(info)
        hybrid_present = _hybrid_path_arrays_present(info)
        if not isinstance(viewpoints, list) or not viewpoints or not slope_present or not hybrid_present:
            transition_missing += 1
        if not hybrid_present:
            hybrid_transition_missing += 1
            continue
        if not slope_present:
            continue
        expected = len(viewpoints)
        for mask_name in ("action_mask", "sampling_mask", "hard_risk_clean_mask"):
            mask = info.get(mask_name)
            if not isinstance(mask, list) or len(mask) != expected:
                mask_mismatch += 1
        if isinstance(info.get("action_mask"), list) and isinstance(info.get("hard_risk_clean_mask"), list) and isinstance(info.get("sampling_mask"), list):
            hybrid_reachable = info.get("hybrid_astar_reachable_flags")
            if not isinstance(hybrid_reachable, list) or len(hybrid_reachable) != len(info["sampling_mask"]):
                hybrid_reachable = [True for _ in info["sampling_mask"]]
            for action_allowed, hard_clean, hybrid_ok, sampled in zip(
                info["action_mask"],
                info["hard_risk_clean_mask"],
                hybrid_reachable,
                info["sampling_mask"],
            ):
                if sampled is not (action_allowed is True and hard_clean is True and hybrid_ok is True):
                    sampling_contract_mismatch += 1
        if action_index is None or action_index < 0 or action_index >= expected:
            binding_mismatch += 1
        else:
            if list(viewpoints[action_index]) != list(info.get("selected_viewpoint") or []):
                binding_mismatch += 1
            if _int_or_none(info.get("selected_theta_deg")) != _theta_from_viewpoint(info.get("selected_viewpoint")):
                binding_mismatch += 1
            if _mask_allows(info, "action_mask", action_index) is not True:
                mask_violation += 1
            if _mask_allows(info, "sampling_mask", action_index) is not True:
                mask_violation += 1
            if _mask_allows(info, "hard_risk_clean_mask", action_index) is not True:
                mask_violation += 1
            selected_hybrid = _selected_hybrid_path(info, action_index)
            if selected_hybrid is None:
                hybrid_binding_mismatch += 1
            else:
                if _finite_float(selected_hybrid["hybrid_astar_path_cost"]) is None:
                    hybrid_binding_mismatch += 1
                if not isinstance(selected_hybrid["hybrid_astar_pose_path_hash"], str) or not selected_hybrid["hybrid_astar_pose_path_hash"]:
                    hybrid_binding_mismatch += 1
                if selected_hybrid["hybrid_astar_trajectory_kind"] != "hybrid_astar_pose_path":
                    hybrid_binding_mismatch += 1
                if _finite_float(selected_hybrid["legacy_grid_astar_path_cost"]) is None:
                    hybrid_binding_mismatch += 1
                if _finite_float(selected_hybrid["hybrid_vs_grid_path_cost_delta"]) is None:
                    hybrid_binding_mismatch += 1
                if selected_hybrid["default_astar_replaced"] is True:
                    hybrid_binding_mismatch += 1
                if selected_hybrid["hybrid_astar_ackermann_feasible_claimed"] is True:
                    hybrid_binding_mismatch += 1
        if info.get("strict_obstacle_aware_new_visible_cell_count") is not True:
            strict_false += 1
        if info.get("default_astar_replaced") is True or any(value is True for value in _list_or_empty(info.get("default_astar_replaced_flags"))):
            default_astar_replaced_count += 1
        if info.get("hybrid_astar_ackermann_feasible_claimed") is True or any(value is True for value in _list_or_empty(info.get("hybrid_astar_ackermann_feasible_claimed_flags"))):
            ackermann_claimed_count += 1
        if info.get("hard_risk_violation") is True:
            hard_risk += 1
        if info.get("open_grid_fallback_used") is True:
            open_grid += 1

    reward_missing = sum(1 for row in rewards if not _row_has_slope_reward(row))
    hybrid_reward_missing = sum(1 for row in rewards if not _row_has_hybrid_reward(row))
    reward_fallback = sum(
        1
        for row in rewards
        if row.get("point_only_reward_fallback_used") is True
        or row.get("unobstructed_theta_reward_fallback_used") is True
    )
    grid_fallback = sum(1 for row in rewards if row.get("point_grid_path_cost_fallback_used") is True)
    transition_by_id = {str(row.get("transition_id")): row for row in transitions if row.get("transition_id") is not None}
    transition_ids = {str(row.get("transition_id")) for row in transitions if str(row.get("transition_id") or "").strip()}
    reward_ids = {str(row.get("transition_id")) for row in rewards if str(row.get("transition_id") or "").strip()}
    batch_ids = {str(row.get("transition_id")) for row in batch_rows if str(row.get("transition_id") or "").strip()}
    blank_transition_id_count = sum(1 for row in transitions if not str(row.get("transition_id") or "").strip())
    blank_reward_transition_id_count = sum(1 for row in rewards if not str(row.get("transition_id") or "").strip())
    blank_batch_transition_id_count = sum(1 for row in batch_rows if not str(row.get("transition_id") or "").strip())
    reward_binding_mismatch = 0
    reward_hybrid_binding_mismatch = 0
    for row in rewards:
        transition = transition_by_id.get(str(row.get("transition_id")))
        if transition is None:
            reward_binding_mismatch += 1
            continue
        info = transition.get("info") if isinstance(transition.get("info"), dict) else {}
        action_index = _int_or_none(transition.get("action_index"))
        if list(row.get("candidate_viewpoint") or []) != list(info.get("selected_viewpoint") or []):
            reward_binding_mismatch += 1
        if _int_or_none(row.get("candidate_theta_deg")) != _int_or_none(info.get("selected_theta_deg")):
            reward_binding_mismatch += 1
        selected = _selected_slope_footprint(info, action_index)
        if selected is None:
            reward_binding_mismatch += 1
            continue
        if _finite_float(row.get("obstacle_aware_new_visible_cell_count")) != _finite_float(selected["count"]):
            reward_binding_mismatch += 1
        if row.get("obstacle_aware_theta_coverage_hash") != selected["hash"]:
            reward_binding_mismatch += 1
        selected_hybrid = _selected_hybrid_path(info, action_index)
        if selected_hybrid is None:
            reward_hybrid_binding_mismatch += 1
            continue
        if row.get("path_cost_source") != HYBRID_ASTAR_PATH_COST_SOURCE:
            reward_hybrid_binding_mismatch += 1
        if _finite_float(row.get("hybrid_astar_path_cost")) != _finite_float(selected_hybrid["hybrid_astar_path_cost"]):
            reward_hybrid_binding_mismatch += 1
        if row.get("hybrid_astar_pose_path_hash") != selected_hybrid["hybrid_astar_pose_path_hash"]:
            reward_hybrid_binding_mismatch += 1
        if row.get("hybrid_astar_trajectory_kind") != selected_hybrid["hybrid_astar_trajectory_kind"]:
            reward_hybrid_binding_mismatch += 1
        if _finite_float(row.get("legacy_grid_astar_path_cost")) != _finite_float(selected_hybrid["legacy_grid_astar_path_cost"]):
            reward_hybrid_binding_mismatch += 1
        if _finite_float(row.get("hybrid_vs_grid_path_cost_delta")) != _finite_float(selected_hybrid["hybrid_vs_grid_path_cost_delta"]):
            reward_hybrid_binding_mismatch += 1

    actual_source_roi_expansion_root = _manifest_source_roi_expansion_root(stage21_1_manifest)
    source_root_match = _same_path(expected_source_roi_expansion_root, actual_source_roi_expansion_root)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "transition_count": len(transitions),
        "reward_row_count": len(rewards),
        "batch_row_count": len(batch_rows),
        "transition_id_count": len(transition_ids),
        "reward_transition_id_count": len(reward_ids),
        "batch_transition_id_count": len(batch_ids),
        "blank_transition_id_count": blank_transition_id_count,
        "blank_reward_transition_id_count": blank_reward_transition_id_count,
        "blank_batch_transition_id_count": blank_batch_transition_id_count,
        "transition_missing_reward_count": len(transition_ids - reward_ids),
        "reward_transition_id_not_in_stage21_1_count": len(reward_ids - transition_ids),
        "batch_transition_id_not_in_stage21_1_count": len(batch_ids - transition_ids),
        "batch_transition_id_not_in_stage21_2_count": len(batch_ids - reward_ids),
        "transition_missing_batch_count": len(transition_ids - batch_ids),
        "reward_missing_batch_count": len(reward_ids - batch_ids),
        "stage21_1_expected_source_roi_expansion_root": expected_source_roi_expansion_root,
        "stage21_1_actual_source_roi_expansion_root": actual_source_roi_expansion_root,
        "stage21_1_source_roi_expansion_root_match": source_root_match,
        "slope_theta_transition_contract_missing_count": transition_missing,
        "strict_obstacle_aware_new_visible_false_count": strict_false,
        "mask_length_mismatch_count": mask_mismatch,
        "sampling_mask_contract_mismatch_count": sampling_contract_mismatch,
        "action_viewpoint_binding_mismatch_count": binding_mismatch,
        "action_viewpoint_path_cost_binding_mismatch_count": hybrid_binding_mismatch,
        "hard_risk_violation_count": hard_risk,
        "mask_violation_count": mask_violation,
        "path_planning_failure_count": sum(1 for row in rejections if "path_planning" in str(row.get("reason") or row.get("reason_code") or "")),
        "open_grid_fallback_count": open_grid,
        "slope_theta_reward_contract_missing_count": reward_missing,
        "hybrid_path_reward_contract_missing_count": hybrid_reward_missing,
        "reward_fallback_used_count": reward_fallback,
        "point_grid_path_cost_fallback_used_count": grid_fallback,
        "reward_slope_footprint_binding_mismatch_count": reward_binding_mismatch,
        "reward_hybrid_path_binding_mismatch_count": reward_hybrid_binding_mismatch,
        "slope_theta_batch_reward_contract_missing_count": (
            stage21_3._slope_obstacle_reward_contract_missing_count(batch_rows) if batch_rows else 0
        ),
        "hybrid_path_batch_contract_missing_count": (
            stage21_3._hybrid_astar_path_cost_contract_missing_count(batch_rows) if batch_rows else 0
        ),
        "stage21_3_rejects_old_reward_batch": stage21_3._slope_obstacle_reward_contract_missing_count([{"reward": 0.0}]) == 1,
        "stage21_3_rejects_grid_only_path_cost_batch": stage21_3._hybrid_astar_path_cost_contract_missing_count([{"reward": 0.0}]) == 1,
        "hybrid_path_transition_contract_missing_count": hybrid_transition_missing,
        "default_astar_replaced_count": default_astar_replaced_count,
        "ackermann_feasible_claimed_count": ackermann_claimed_count,
        "default_astar_replaced": default_astar_replaced_count > 0,
        "ackermann_feasible_claimed": ackermann_claimed_count > 0,
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
        return "failed", ROUTE_BOUNDARY, "stage24_3_boundary_rejected"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage24_3_inputs_missing_or_untrusted"
    if (
        stage21_1_summary.get("status") != "passed"
        or audit["transition_count"] <= 0
        or audit["transition_id_count"] <= 0
        or audit["blank_transition_id_count"] > 0
        or audit["slope_theta_transition_contract_missing_count"] > 0
        or audit["hybrid_path_transition_contract_missing_count"] > 0
        or audit["strict_obstacle_aware_new_visible_false_count"] > 0
        or audit["stage21_1_source_roi_expansion_root_match"] is not True
    ):
        return "failed", ROUTE_COLLECTOR, "stage21_1_hybrid_path_cost_transition_contract_missing"
    if (
        audit["mask_length_mismatch_count"]
        or audit["sampling_mask_contract_mismatch_count"]
        or audit["action_viewpoint_binding_mismatch_count"]
        or audit["action_viewpoint_path_cost_binding_mismatch_count"]
        or audit["default_astar_replaced_count"]
        or audit["ackermann_feasible_claimed_count"]
        or audit["mask_violation_count"]
        or audit["hard_risk_violation_count"]
        or audit["path_planning_failure_count"]
        or audit["open_grid_fallback_count"]
    ):
        return "failed", ROUTE_BINDING, "stage21_1_action_viewpoint_or_safety_binding_failed"
    if (
        stage21_2_summary.get("status") != "passed"
        or audit["reward_row_count"] <= 0
        or audit["reward_transition_id_count"] <= 0
        or audit["blank_reward_transition_id_count"]
        or audit["transition_missing_reward_count"]
        or audit["reward_transition_id_not_in_stage21_1_count"]
        or audit["slope_theta_reward_contract_missing_count"]
        or audit["hybrid_path_reward_contract_missing_count"]
        or audit["reward_fallback_used_count"]
        or audit["point_grid_path_cost_fallback_used_count"]
        or audit["reward_slope_footprint_binding_mismatch_count"]
        or audit["reward_hybrid_path_binding_mismatch_count"]
    ):
        return "failed", ROUTE_REWARD, "stage21_2_hybrid_path_cost_reward_provenance_missing"
    if (
        stage21_3_summary.get("status") != "passed"
        or audit["batch_row_count"] <= 0
        or audit["batch_transition_id_count"] <= 0
        or audit["blank_batch_transition_id_count"]
        or audit["batch_transition_id_not_in_stage21_1_count"]
        or audit["batch_transition_id_not_in_stage21_2_count"]
        or audit["transition_missing_batch_count"]
        or audit["reward_missing_batch_count"]
        or audit["hybrid_path_batch_contract_missing_count"]
        or not audit["stage21_3_rejects_old_reward_batch"]
        or not audit["stage21_3_rejects_grid_only_path_cost_batch"]
    ):
        return "failed", ROUTE_BATCH, "stage21_3_hybrid_path_cost_batch_contract_missing"
    return "passed", ROUTE_STAGE24_4, "hybrid_astar_path_cost_collector_reward_batch_ready"


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
        if audit["stage21_1_source_roi_expansion_root_match"] is not True:
            reasons.append("stage21_1_source_roi_expansion_root_mismatch")
        for key in ("transition_count", "transition_id_count", "blank_transition_id_count", "slope_theta_transition_contract_missing_count", "hybrid_path_transition_contract_missing_count", "strict_obstacle_aware_new_visible_false_count"):
            if key == "transition_count" and audit[key] <= 0:
                reasons.append("stage21_1_transition_rows_missing")
            elif key == "transition_id_count" and audit[key] <= 0:
                reasons.append("stage21_1_transition_ids_missing")
            elif key not in {"transition_count", "transition_id_count"} and audit[key]:
                reasons.append(key.removesuffix("_count"))
    if route == ROUTE_BINDING:
        for key in ("mask_length_mismatch_count", "sampling_mask_contract_mismatch_count", "action_viewpoint_binding_mismatch_count", "action_viewpoint_path_cost_binding_mismatch_count", "default_astar_replaced_count", "ackermann_feasible_claimed_count", "mask_violation_count", "hard_risk_violation_count", "path_planning_failure_count", "open_grid_fallback_count"):
            if audit[key]:
                reasons.append(key.removesuffix("_count"))
    if route == ROUTE_REWARD:
        if stage21_2_summary.get("status") != "passed":
            reasons.append("stage21_2_not_passed")
        for key in ("reward_row_count", "reward_transition_id_count", "blank_reward_transition_id_count", "transition_missing_reward_count", "reward_transition_id_not_in_stage21_1_count", "slope_theta_reward_contract_missing_count", "hybrid_path_reward_contract_missing_count", "reward_fallback_used_count", "point_grid_path_cost_fallback_used_count", "reward_slope_footprint_binding_mismatch_count", "reward_hybrid_path_binding_mismatch_count"):
            if key == "reward_row_count" and audit[key] <= 0:
                reasons.append("stage21_2_reward_rows_missing")
            elif key == "reward_transition_id_count" and audit[key] <= 0:
                reasons.append("stage21_2_reward_transition_ids_missing")
            elif key not in {"reward_row_count", "reward_transition_id_count"} and audit[key]:
                reasons.append(key.removesuffix("_count"))
    if route == ROUTE_BATCH:
        if stage21_3_summary.get("status") != "passed":
            reasons.append("stage21_3_not_passed")
        if audit["batch_row_count"] <= 0:
            reasons.append("stage21_3_batch_rows_missing")
        if audit["batch_transition_id_count"] <= 0:
            reasons.append("stage21_3_batch_transition_ids_missing")
        for key in (
            "blank_batch_transition_id_count",
            "batch_transition_id_not_in_stage21_1_count",
            "batch_transition_id_not_in_stage21_2_count",
            "transition_missing_batch_count",
            "reward_missing_batch_count",
        ):
            if audit[key]:
                reasons.append(key.removesuffix("_count"))
        if audit["hybrid_path_batch_contract_missing_count"]:
            reasons.append("hybrid_path_batch_contract_missing")
        if not audit["stage21_3_rejects_old_reward_batch"]:
            reasons.append("stage21_3_old_reward_gate_missing")
        if not audit["stage21_3_rejects_grid_only_path_cost_batch"]:
            reasons.append("stage21_3_grid_only_path_cost_gate_missing")
    return reasons


def _summary_counts(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "trainable_transition_count": audit["transition_count"],
        "reward_evaluation_row_count": audit["reward_row_count"],
        "batch_row_count": audit["batch_row_count"],
        "transition_id_count": audit["transition_id_count"],
        "reward_transition_id_count": audit["reward_transition_id_count"],
        "batch_transition_id_count": audit["batch_transition_id_count"],
        "transition_missing_reward_count": audit["transition_missing_reward_count"],
        "reward_transition_id_not_in_stage21_1_count": audit["reward_transition_id_not_in_stage21_1_count"],
        "batch_transition_id_not_in_stage21_1_count": audit["batch_transition_id_not_in_stage21_1_count"],
        "batch_transition_id_not_in_stage21_2_count": audit["batch_transition_id_not_in_stage21_2_count"],
        "transition_missing_batch_count": audit["transition_missing_batch_count"],
        "reward_missing_batch_count": audit["reward_missing_batch_count"],
        "stage21_1_source_roi_expansion_root_match": audit["stage21_1_source_roi_expansion_root_match"],
        "slope_theta_transition_contract_missing_count": audit["slope_theta_transition_contract_missing_count"],
        "hybrid_path_transition_contract_missing_count": audit["hybrid_path_transition_contract_missing_count"],
        "strict_obstacle_aware_new_visible_false_count": audit["strict_obstacle_aware_new_visible_false_count"],
        "mask_length_mismatch_count": audit["mask_length_mismatch_count"],
        "sampling_mask_contract_mismatch_count": audit["sampling_mask_contract_mismatch_count"],
        "action_viewpoint_binding_mismatch_count": audit["action_viewpoint_binding_mismatch_count"],
        "action_viewpoint_path_cost_binding_mismatch_count": audit["action_viewpoint_path_cost_binding_mismatch_count"],
        "slope_theta_reward_contract_missing_count": audit["slope_theta_reward_contract_missing_count"],
        "hybrid_path_reward_contract_missing_count": audit["hybrid_path_reward_contract_missing_count"],
        "reward_fallback_used_count": audit["reward_fallback_used_count"],
        "point_grid_path_cost_fallback_used_count": audit["point_grid_path_cost_fallback_used_count"],
        "reward_slope_footprint_binding_mismatch_count": audit["reward_slope_footprint_binding_mismatch_count"],
        "reward_hybrid_path_binding_mismatch_count": audit["reward_hybrid_path_binding_mismatch_count"],
        "slope_theta_batch_reward_contract_missing_count": audit["slope_theta_batch_reward_contract_missing_count"],
        "hybrid_path_batch_contract_missing_count": audit["hybrid_path_batch_contract_missing_count"],
        "hard_risk_violation_count": audit["hard_risk_violation_count"],
        "mask_violation_count": audit["mask_violation_count"],
        "path_planning_failure_count": audit["path_planning_failure_count"],
        "open_grid_fallback_count": audit["open_grid_fallback_count"],
        "stage21_3_rejects_old_reward_batch": audit["stage21_3_rejects_old_reward_batch"],
        "stage21_3_rejects_grid_only_path_cost_batch": audit["stage21_3_rejects_grid_only_path_cost_batch"],
        "default_astar_replaced": audit["default_astar_replaced"],
        "ackermann_feasible_claimed": audit["ackermann_feasible_claimed"],
    }


def _slope_arrays_present(info: dict[str, Any]) -> bool:
    viewpoints = info.get("candidate_viewpoints")
    if not isinstance(viewpoints, list) or not viewpoints:
        return False
    expected = len(viewpoints)
    for key in (
        "candidate_theta_deg",
        "obstacle_aware_new_visible_cell_counts",
        "obstacle_aware_theta_coverage_hashes",
        "obstacle_aware_theta_coverage_gain_per_path_costs",
        "slope_obstacle_source_hashes",
        "platform_contract_hashes",
    ):
        value = info.get(key)
        if not isinstance(value, list) or len(value) != expected:
            return False
    return (
        bool(str(info.get("slope_obstacle_source_hash") or "").strip())
        and bool(str(info.get("platform_contract_hash") or "").strip())
        and _finite_float(info.get("max_traversable_slope_deg")) == 30.0
        and info.get("slope_blocked_source_kind") == "slope_blocked_as_obstacle_proxy"
        and info.get("selected_viewpoint") is not None
        and info.get("selected_theta_deg") is not None
    )


def _row_has_slope_reward(row: dict[str, Any]) -> bool:
    return (
        row.get("theta_aware_reward_contract") is True
        and row.get("slope_obstacle_aware_theta_reward_contract") is True
        and row.get("coverage_source") == SLOPE_COVERAGE_SOURCE
        and _finite_float(row.get("obstacle_aware_new_visible_cell_count")) is not None
        and isinstance(row.get("obstacle_aware_theta_coverage_hash"), str)
        and bool(str(row.get("obstacle_aware_theta_coverage_hash")).strip())
        and isinstance(row.get("slope_obstacle_source_hash"), str)
        and bool(str(row.get("slope_obstacle_source_hash")).strip())
        and isinstance(row.get("platform_contract_hash"), str)
        and bool(str(row.get("platform_contract_hash")).strip())
        and _finite_float(row.get("max_traversable_slope_deg")) == 30.0
        and row.get("slope_blocked_source_kind") == "slope_blocked_as_obstacle_proxy"
        and row.get("point_only_reward_fallback_used") is False
        and row.get("unobstructed_theta_reward_fallback_used") is False
    )


def _hybrid_path_arrays_present(info: dict[str, Any]) -> bool:
    viewpoints = info.get("candidate_viewpoints")
    if not isinstance(viewpoints, list) or not viewpoints:
        return False
    expected = len(viewpoints)
    for key in (
        "path_cost_sources",
        "hybrid_astar_path_costs",
        "hybrid_astar_pose_path_hashes",
        "hybrid_astar_trajectory_kinds",
        "legacy_grid_astar_path_costs",
        "hybrid_vs_grid_path_cost_deltas",
        "default_astar_replaced_flags",
        "hybrid_astar_ackermann_feasible_claimed_flags",
    ):
        value = info.get(key)
        if not isinstance(value, list) or len(value) != expected:
            return False
    return (
        info.get("path_cost_source") == HYBRID_ASTAR_PATH_COST_SOURCE
        and info.get("default_astar_replaced") is False
        and info.get("hybrid_astar_ackermann_feasible_claimed") is False
    )


def _row_has_hybrid_reward(row: dict[str, Any]) -> bool:
    if not (
        row.get("hybrid_astar_path_cost_reward_contract") is True
        and row.get("path_cost_source") == HYBRID_ASTAR_PATH_COST_SOURCE
        and _finite_float(row.get("hybrid_astar_path_cost")) is not None
        and _finite_float(row.get("hybrid_astar_path_cost")) > 0.0
        and isinstance(row.get("hybrid_astar_pose_path_hash"), str)
        and bool(str(row.get("hybrid_astar_pose_path_hash")).strip())
        and row.get("hybrid_astar_trajectory_kind") == "hybrid_astar_pose_path"
        and _finite_float(row.get("legacy_grid_astar_path_cost")) is not None
        and _finite_float(row.get("hybrid_vs_grid_path_cost_delta")) is not None
        and row.get("default_astar_replaced") is False
        and row.get("hybrid_astar_ackermann_feasible_claimed") is False
        and row.get("point_grid_path_cost_fallback_used") is False
    ):
        return False
    metrics = row.get("reward_metrics") if isinstance(row.get("reward_metrics"), dict) else row.get("metrics")
    if not isinstance(metrics, dict):
        return False
    return _finite_float(metrics.get("path_cost_m")) == _finite_float(row.get("hybrid_astar_path_cost"))


def _selected_slope_footprint(info: dict[str, Any], action_index: int | None) -> dict[str, Any] | None:
    if action_index is None or action_index < 0:
        return None
    counts = info.get("obstacle_aware_new_visible_cell_counts")
    hashes = info.get("obstacle_aware_theta_coverage_hashes")
    gains = info.get("obstacle_aware_theta_coverage_gain_per_path_costs")
    if not isinstance(counts, list) or not isinstance(hashes, list) or not isinstance(gains, list):
        return None
    if action_index >= len(counts) or action_index >= len(hashes) or action_index >= len(gains):
        return None
    return {"count": counts[action_index], "hash": hashes[action_index], "gain": gains[action_index]}


def _selected_hybrid_path(info: dict[str, Any], action_index: int | None) -> dict[str, Any] | None:
    if action_index is None or action_index < 0:
        return None
    sources = info.get("path_cost_sources")
    costs = info.get("hybrid_astar_path_costs")
    hashes = info.get("hybrid_astar_pose_path_hashes")
    kinds = info.get("hybrid_astar_trajectory_kinds")
    legacy_costs = info.get("legacy_grid_astar_path_costs")
    deltas = info.get("hybrid_vs_grid_path_cost_deltas")
    default_flags = info.get("default_astar_replaced_flags")
    ackermann_flags = info.get("hybrid_astar_ackermann_feasible_claimed_flags")
    arrays = (sources, costs, hashes, kinds, legacy_costs, deltas, default_flags, ackermann_flags)
    if not all(isinstance(value, list) for value in arrays):
        return None
    if any(action_index >= len(value) for value in arrays if isinstance(value, list)):
        return None
    if sources[action_index] != HYBRID_ASTAR_PATH_COST_SOURCE:
        return None
    return {
        "hybrid_astar_path_cost": costs[action_index],
        "hybrid_astar_pose_path_hash": hashes[action_index],
        "hybrid_astar_trajectory_kind": kinds[action_index],
        "legacy_grid_astar_path_cost": legacy_costs[action_index],
        "hybrid_vs_grid_path_cost_delta": deltas[action_index],
        "default_astar_replaced": default_flags[action_index],
        "hybrid_astar_ackermann_feasible_claimed": ackermann_flags[action_index],
    }


def _mask_allows(info: dict[str, Any], mask_name: str, action_index: int | None) -> bool | None:
    if action_index is None or action_index < 0:
        return None
    mask = info.get(mask_name)
    if not isinstance(mask, list) or action_index >= len(mask):
        return None
    return mask[action_index] is True


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _theta_from_viewpoint(value: Any) -> int | None:
    if isinstance(value, list) and len(value) >= 3:
        return _int_or_none(value[2])
    return None


def _input_rejections(config: dict[str, Any]) -> list[str]:
    summary_path = Path(config["stage24_2_root"]) / "xunce-stage24-2-summary.json"
    if not summary_path.is_file():
        return ["missing_stage24_2_summary"]
    summary = _read_json(summary_path)
    reasons: list[str] = []
    if summary.get("status") != "passed":
        reasons.append("stage24_2_not_passed")
    if summary.get("next_required_change") != ROUTE_FROM_STAGE24_2:
        reasons.append("stage24_2_route_not_stage24_3")
    if summary.get("coverage_source") != SLOPE_COVERAGE_SOURCE:
        reasons.append("stage24_2_coverage_source_not_slope_theta_los")
    if summary.get("path_cost_source") != HYBRID_ASTAR_PATH_COST_SOURCE:
        reasons.append("stage24_2_path_cost_source_not_hybrid_astar")
    for key in ("hybrid_astar_path_cost_contract_missing_count", "point_grid_path_cost_fallback_used_count"):
        if int(summary.get(key, 0) or 0) != 0:
            reasons.append(f"stage24_2_{key}")
    if float(summary.get("max_traversable_slope_deg") or 0.0) != 30.0:
        reasons.append("stage24_2_max_traversable_slope_not_30")
    if summary.get("default_astar_replaced") is not False:
        reasons.append("stage24_2_default_astar_replaced")
    if summary.get("ackermann_feasible_claimed") is not False:
        reasons.append("stage24_2_ackermann_feasible_claimed")
    return reasons


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
        "stage24_2_root": DEFAULT_STAGE24_2_ROOT,
        "stage21_1_base_config": "configs/xunce_stage21_1_on_policy_ppo_rollout_collector_v1.json",
        "stage21_2_base_config": "configs/xunce_stage21_2_coverage_first_ppo_reward_contract_v1.json",
        "stage21_3_base_config": "configs/xunce_stage21_3_ppo_batch_validation_v1.json",
        "coverage_first_reward_profile": "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json",
        "required_scenario_count": 2,
        "rollout_steps": 4,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "min_trainable_transition_count": 1,
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_model_id": "theta-fov-90-range-radius/v1",
        "sensor_fov_deg": 90.0,
        "sensor_range_cells": 3,
        "theta_coverage_denominator_cells": 100.0,
        "max_traversable_slope_deg": 30.0,
        "stage24_3_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    for key in ("stage24_2_root", "stage21_1_base_config", "stage21_2_base_config", "stage21_3_base_config", "coverage_first_reward_profile"):
        config[key] = str(_resolve_path(Path(str(config[key])), repo_root))
    for key in ("required_scenario_count", "rollout_steps", "dynamic_max_candidates_per_step", "dynamic_proposal_pool_limit_per_step", "min_trainable_transition_count", "theta_bin_count", "theta_step_deg", "sensor_range_cells"):
        config[key] = _positive_int(config[key], key)
    for key in ("sensor_fov_deg", "theta_coverage_denominator_cells", "max_traversable_slope_deg"):
        config[key] = _positive_float(config[key], key)
    for field in BOUNDARY_FIELDS:
        config.setdefault(field, False)
    return config


def _read_stage24_2_summary(config: dict[str, Any]) -> dict[str, Any]:
    return _read_json(Path(config["stage24_2_root"]) / "xunce-stage24-2-summary.json")


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage24.3 Hybrid A* Path Cost PPO Collector Smoke",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- trainable_transition_count: `{summary['trainable_transition_count']}`",
            f"- slope_theta_transition_contract_missing_count: `{summary['slope_theta_transition_contract_missing_count']}`",
            f"- hybrid_path_transition_contract_missing_count: `{summary['hybrid_path_transition_contract_missing_count']}`",
            f"- strict_obstacle_aware_new_visible_false_count: `{summary['strict_obstacle_aware_new_visible_false_count']}`",
            f"- slope_theta_reward_contract_missing_count: `{summary['slope_theta_reward_contract_missing_count']}`",
            f"- hybrid_path_reward_contract_missing_count: `{summary['hybrid_path_reward_contract_missing_count']}`",
            f"- hybrid_path_batch_contract_missing_count: `{summary['hybrid_path_batch_contract_missing_count']}`",
            "",
            "Stage24.3 validates collector/reward/batch contracts only. It does not run PPO update, publish checkpoints, replace default policy, connect an executor, or start canary traffic.",
        ]
    ) + "\n"


def _load_json_template(path: str, repo_root: Path) -> dict[str, Any]:
    return _read_json(_resolve_path(Path(path), repo_root))


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _manifest_source_roi_expansion_root(manifest: dict[str, Any]) -> str | None:
    model_audit = manifest.get("model_audit")
    if not isinstance(model_audit, dict):
        return None
    value = model_audit.get("source_roi_expansion_root")
    return str(value) if isinstance(value, str) and value.strip() else None


def _same_path(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    try:
        return str(Path(left).resolve()).lower() == str(Path(right).resolve()).lower()
    except OSError:
        return str(left).replace("\\", "/").rstrip("/").lower() == str(right).replace("\\", "/").rstrip("/").lower()


def _positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _positive_float(value: Any, name: str) -> float:
    number = _finite_float(value)
    if number is None or number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


if __name__ == "__main__":
    raise SystemExit(main())

