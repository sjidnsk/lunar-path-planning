from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
PATH_PLANNER_SRC = REPO_ROOT / "path-planner" / "src"
for _path in (SCRIPTS_ROOT, PATH_PLANNER_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from path_planner.core import Cell  # noqa: E402
from xunce_hybrid_astar_candidate_path_cost import (  # noqa: E402
    PATH_COST_SOURCE,
    build_cost_grid_from_sidecar,
    evaluate_hybrid_astar_candidate_path_cost,
)
from xunce_obstacle_aware_theta_sensor_coverage import (  # noqa: E402
    visible_cells_for_viewpoint_with_obstacles,
)
from xunce_synthetic_terrain_features import (  # noqa: E402
    SYNTHETIC_SOURCE_KIND,
    SYNTHETIC_TERRAIN_MODEL_ID,
    augmented_sidecar_with_synthetic_source,
    generate_synthetic_terrain_augmentation,
)


STAGE_ID = "xunce-stage26-0-synthetic-rock-pit-terrain-augmentation-contract"
CONFIG_SCHEMA_VERSION = "xunce-stage26-0-synthetic-rock-pit-terrain-augmentation-contract-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-0-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-0-manifest/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-0-next-stage-routing/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract_v1"
)

SUMMARY_FILE = "xunce-stage26-0-summary.json"
FEATURE_CATALOG_FILE = "xunce-stage26-0-synthetic-feature-catalog.jsonl"
OBSTACLE_SOURCE_FILE = "xunce-stage26-0-synthetic-obstacle-source.json"
MAP_AUDIT_FILE = "xunce-stage26-0-map-augmentation-audit.json"
LOS_AUDIT_FILE = "xunce-stage26-0-los-impact-audit.json"
HYBRID_AUDIT_FILE = "xunce-stage26-0-hybrid-path-impact-audit.json"
ROUTING_FILE = "xunce-stage26-0-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-0-report.md"
MANIFEST_FILE = "xunce-stage26-0-manifest.json"

ROUTE_INPUTS = "rerun_stage26_0_required_map_inputs"
ROUTE_HASH = "repair_stage26_0_synthetic_feature_hash_contract"
ROUTE_SOURCE = "repair_stage26_0_synthetic_source_semantics"
ROUTE_CONNECTIVITY = "repair_stage26_0_synthetic_connectivity_constraints"
ROUTE_LOS_DENSITY = "tune_stage26_synthetic_los_blocker_density"
ROUTE_HYBRID_DENSITY = "tune_stage26_synthetic_obstacle_density"
ROUTE_STAGE26_1 = "run_stage26_1_synthetic_terrain_collector_smoke"
ROUTE_BOUNDARY = "resolve_stage26_0_boundary_rejections"

BOUNDARY_FIELDS = (
    "stage26_0_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)

COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.0 synthetic rock/pit terrain augmentation contract.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "augmented_sidecars").mkdir(parents=True, exist_ok=True)

    boundary_reasons = _boundary_rejections(config)
    input_reasons, sidecar_paths = _input_rejections(config, repo_root)

    catalog_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    sidecar_audits: list[dict[str, Any]] = []
    los_rows: list[dict[str, Any]] = []
    hybrid_rows: list[dict[str, Any]] = []
    augmented_sidecar_paths: list[str] = []

    if not boundary_reasons and not input_reasons:
        for index, sidecar_path in enumerate(sidecar_paths):
            sidecar = _read_json(sidecar_path)
            width, height = _sidecar_dimensions(sidecar)
            resolution_m = _sidecar_resolution_m(sidecar)
            start_cell = _sidecar_start_cell(sidecar, config)
            existing_hard = _existing_hard_cells(sidecar)
            sidecar_config = dict(config)
            sidecar_config["synthetic_terrain_seed"] = int(config["synthetic_terrain_seed"]) + index
            augmentation = generate_synthetic_terrain_augmentation(
                width=width,
                height=height,
                resolution_m=resolution_m,
                start_cell=start_cell,
                existing_hard_obstacle_cells=existing_hard,
                config=sidecar_config,
            )
            source = augmentation["source"]
            synthetic_source_id = f"synthetic-source-{index:03d}"
            source.update(
                {
                    "synthetic_source_id": synthetic_source_id,
                    "source_sidecar": str(sidecar_path),
                    "platform_contract_hash": sidecar.get("platform_contract_hash") or config.get("platform_contract_hash"),
                    "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
                    "coverage_source": COVERAGE_SOURCE,
                    "path_cost_source": PATH_COST_SOURCE,
                }
            )
            source_rows.append(source)
            for row in augmentation["catalog"]:
                catalog_rows.append({**row, "synthetic_source_id": synthetic_source_id, "source_sidecar": str(sidecar_path)})
            sidecar_audits.append(
                {
                    **augmentation["audit"],
                    "synthetic_source_id": synthetic_source_id,
                    "source_sidecar": str(sidecar_path),
                    "width": width,
                    "height": height,
                    "resolution_m": resolution_m,
                    "platform_contract_hash": source["platform_contract_hash"],
                    "max_traversable_slope_deg": source["max_traversable_slope_deg"],
                }
            )
            augmented = augmented_sidecar_with_synthetic_source(sidecar, source)
            augmented_path = output_root / "augmented_sidecars" / f"stage26_0_{index:03d}.synthetic-sidecar.json"
            _write_json(augmented_path, augmented)
            augmented_sidecar_paths.append(str(augmented_path))
            los_rows.extend(_los_impact_rows(sidecar, augmented, source, config, index=index))
            hybrid_rows.extend(_hybrid_impact_rows(sidecar, augmented, source, config, index=index))

    source_aggregate = _aggregate_sources(source_rows)
    map_audit = _map_audit(sidecar_audits, source_aggregate)
    los_audit = _los_audit(los_rows)
    hybrid_audit = _hybrid_audit(hybrid_rows)
    status, route, route_reason = _route(boundary_reasons, input_reasons, map_audit, source_aggregate, los_audit, hybrid_audit)
    blocking = [*boundary_reasons, *input_reasons, *_audit_blockers(map_audit, source_aggregate, los_audit, hybrid_audit)]

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking,
        "source_sidecar_count": len(sidecar_paths),
        "synthetic_terrain_model_id": SYNTHETIC_TERRAIN_MODEL_ID,
        "synthetic_terrain_seed": int(config["synthetic_terrain_seed"]),
        "synthetic_terrain_hash": source_aggregate.get("synthetic_terrain_hash"),
        "synthetic_rock_count": source_aggregate["synthetic_rock_count"],
        "synthetic_pit_count": source_aggregate["synthetic_pit_count"],
        "synthetic_hard_obstacle_cell_count": source_aggregate["synthetic_hard_obstacle_cell_count"],
        "synthetic_los_blocker_cell_count": source_aggregate["synthetic_los_blocker_cell_count"],
        "synthetic_high_risk_cell_count": source_aggregate["synthetic_high_risk_cell_count"],
        "physical_obstacle_cells_written": source_aggregate["physical_obstacle_cells_written"],
        "source_kind": SYNTHETIC_SOURCE_KIND,
        "blocked_fraction_max": map_audit["blocked_fraction_max"],
        "effective_blocked_fraction_max": map_audit["effective_blocked_fraction_max"],
        "blocked_fraction_passed": map_audit["blocked_fraction_passed"],
        "effective_blocked_fraction_passed": map_audit["effective_blocked_fraction_passed"],
        "connectivity_passed": map_audit["connectivity_passed"],
        "effective_connectivity_passed": map_audit["effective_connectivity_passed"],
        "start_clearance_passed": map_audit["start_clearance_passed"],
        "effective_start_clearance_passed": map_audit["effective_start_clearance_passed"],
        "los_material_viewpoint_count": los_audit["los_material_viewpoint_count"],
        "hybrid_path_cost_material_candidate_count": hybrid_audit["hybrid_path_cost_material_candidate_count"],
        "platform_contract_hashes": sorted(
            {str(row.get("platform_contract_hash")) for row in source_rows if row.get("platform_contract_hash")}
        ),
        "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "augmented_sidecar_paths": augmented_sidecar_paths,
        "stage26_0_authorized": False,
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
        "stage26_0_authorized": False,
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
        "synthetic_feature_catalog": str(output_root / FEATURE_CATALOG_FILE),
        "synthetic_obstacle_source": str(output_root / OBSTACLE_SOURCE_FILE),
        "map_augmentation_audit": str(output_root / MAP_AUDIT_FILE),
        "los_impact_audit": str(output_root / LOS_AUDIT_FILE),
        "hybrid_path_impact_audit": str(output_root / HYBRID_AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
        "augmented_sidecars": augmented_sidecar_paths,
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_jsonl(output_root / FEATURE_CATALOG_FILE, catalog_rows)
    _write_json(output_root / OBSTACLE_SOURCE_FILE, source_aggregate)
    _write_json(output_root / MAP_AUDIT_FILE, map_audit)
    _write_json(output_root / LOS_AUDIT_FILE, los_audit)
    _write_json(output_root / HYBRID_AUDIT_FILE, hybrid_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _los_impact_rows(
    sidecar: dict[str, Any],
    augmented: dict[str, Any],
    source: dict[str, Any],
    config: dict[str, Any],
    *,
    index: int,
) -> list[dict[str, Any]]:
    width, height = _sidecar_dimensions(sidecar)
    sensor_range = int(config.get("sensor_range_cells", 5))
    sensor_fov = float(config.get("sensor_fov_deg", 90.0))
    existing_los = _existing_los_cells(sidecar)
    synthetic_los = _cells(source.get("synthetic_los_blocker_cells"))
    rows: list[dict[str, Any]] = []
    for sample_index, viewpoint in enumerate(_viewpoints_from_blockers(synthetic_los, width, height, limit=int(config.get("los_sample_limit", 64)))):
        cell, theta_deg = viewpoint
        base = visible_cells_for_viewpoint_with_obstacles(
            cell,
            theta_deg=theta_deg,
            sensor_range_cells=sensor_range,
            sensor_fov_deg=sensor_fov,
            obstacle_cells=existing_los,
            bounds=(0, 0, width - 1, height - 1),
        )
        synth = visible_cells_for_viewpoint_with_obstacles(
            cell,
            theta_deg=theta_deg,
            sensor_range_cells=sensor_range,
            sensor_fov_deg=sensor_fov,
            obstacle_cells=existing_los | synthetic_los,
            bounds=(0, 0, width - 1, height - 1),
        )
        rows.append(
            {
                "schema_version": "xunce-stage26-0-los-impact-row/v1",
                "source_index": index,
                "sample_index": sample_index,
                "synthetic_source_id": source["synthetic_source_id"],
                "viewpoint_cell": [cell[0], cell[1]],
                "theta_deg": theta_deg,
                "sensor_range_cells": sensor_range,
                "sensor_fov_deg": sensor_fov,
                "base_visible_cell_count": len(base.visible_cells),
                "synthetic_visible_cell_count": len(synth.visible_cells),
                "base_coverage_hash": base.obstacle_aware_theta_coverage_hash,
                "synthetic_coverage_hash": synth.obstacle_aware_theta_coverage_hash,
                "synthetic_occluded_cell_count": synth.occluded_cell_count,
                "synthetic_blocked_obstacle_cell_count": synth.blocked_by_obstacle_count,
                "los_material": base.obstacle_aware_theta_coverage_hash != synth.obstacle_aware_theta_coverage_hash,
                "synthetic_terrain_hash": source["synthetic_terrain_hash"],
                "coverage_source": COVERAGE_SOURCE,
            }
        )
    return rows


def _hybrid_impact_rows(
    sidecar: dict[str, Any],
    augmented: dict[str, Any],
    source: dict[str, Any],
    config: dict[str, Any],
    *,
    index: int,
) -> list[dict[str, Any]]:
    base_grid = build_cost_grid_from_sidecar(sidecar)
    augmented_grid = build_cost_grid_from_sidecar(augmented)
    width, height = _sidecar_dimensions(sidecar)
    resolution = _sidecar_resolution_m(sidecar)
    platform_hash = str(source.get("platform_contract_hash") or config.get("platform_contract_hash") or "synthetic-stage26-platform-hash")
    max_slope = float(source.get("max_traversable_slope_deg") or config["max_traversable_slope_deg"])
    rows: list[dict[str, Any]] = []
    candidates = _hybrid_candidates_from_hard_obstacles(
        source,
        width,
        height,
        base_grid=base_grid,
        augmented_grid=augmented_grid,
        limit=int(config.get("hybrid_sample_limit", 32)),
    )
    for sample_index, candidate in enumerate(candidates):
        start_cell = candidate["start_cell"]
        current_pose = [(start_cell[0] + 0.5) * resolution, (start_cell[1] + 0.5) * resolution, 0.0]
        base_row = evaluate_hybrid_astar_candidate_path_cost(
            grid=base_grid,
            current_pose=current_pose,
            candidate=candidate["candidate"],
            platform_contract_hash=platform_hash,
            max_traversable_slope_deg=max_slope,
            theta_bin_count=int(config.get("hybrid_astar_theta_bin_count", 72)),
            goal_theta_tolerance_deg=float(config.get("hybrid_astar_goal_theta_tolerance_deg", 5.0)),
            max_iterations=int(config.get("hybrid_astar_max_iterations", 20000)),
            primitive_duration_s=float(config.get("hybrid_astar_primitive_duration_s", 1.5)),
            integration_dt_s=float(config.get("hybrid_astar_integration_dt_s", 0.25)),
            max_speed_mps=float(config.get("hybrid_astar_max_speed_mps", 3.0)),
            max_angular_speed_degps=float(config.get("hybrid_astar_max_angular_speed_degps", 45.0)),
            rotation_cost_weight=float(config.get("hybrid_astar_rotation_cost_weight", 0.2)),
            reverse_penalty_weight=float(config.get("hybrid_astar_reverse_penalty_weight", 0.5)),
            turn_penalty_weight=float(config.get("hybrid_astar_turn_penalty_weight", 0.05)),
        )
        synth_row = evaluate_hybrid_astar_candidate_path_cost(
            grid=augmented_grid,
            current_pose=current_pose,
            candidate=candidate["candidate"],
            platform_contract_hash=platform_hash,
            max_traversable_slope_deg=max_slope,
            theta_bin_count=int(config.get("hybrid_astar_theta_bin_count", 72)),
            goal_theta_tolerance_deg=float(config.get("hybrid_astar_goal_theta_tolerance_deg", 5.0)),
            max_iterations=int(config.get("hybrid_astar_max_iterations", 20000)),
            primitive_duration_s=float(config.get("hybrid_astar_primitive_duration_s", 1.5)),
            integration_dt_s=float(config.get("hybrid_astar_integration_dt_s", 0.25)),
            max_speed_mps=float(config.get("hybrid_astar_max_speed_mps", 3.0)),
            max_angular_speed_degps=float(config.get("hybrid_astar_max_angular_speed_degps", 45.0)),
            rotation_cost_weight=float(config.get("hybrid_astar_rotation_cost_weight", 0.2)),
            reverse_penalty_weight=float(config.get("hybrid_astar_reverse_penalty_weight", 0.5)),
            turn_penalty_weight=float(config.get("hybrid_astar_turn_penalty_weight", 0.05)),
        )
        base_cost = base_row.get("hybrid_astar_path_cost")
        synth_cost = synth_row.get("hybrid_astar_path_cost")
        material = base_row.get("hybrid_astar_reachable") != synth_row.get("hybrid_astar_reachable")
        if base_cost is not None and synth_cost is not None:
            material = material or abs(float(synth_cost) - float(base_cost)) > float(config.get("hybrid_material_cost_delta_threshold", 0.01))
        rows.append(
            {
                "schema_version": "xunce-stage26-0-hybrid-path-impact-row/v1",
                "source_index": index,
                "sample_index": sample_index,
                "synthetic_source_id": source["synthetic_source_id"],
                "start_cell": [start_cell[0], start_cell[1]],
                "candidate_viewpoint": candidate["candidate"]["candidate_viewpoint"],
                "base_hybrid_astar_reachable": base_row.get("hybrid_astar_reachable"),
                "synthetic_hybrid_astar_reachable": synth_row.get("hybrid_astar_reachable"),
                "base_hybrid_astar_path_cost": base_cost,
                "synthetic_hybrid_astar_path_cost": synth_cost,
                "hybrid_path_cost_delta": None if base_cost is None or synth_cost is None else float(synth_cost) - float(base_cost),
                "base_hybrid_astar_failure_reason": base_row.get("hybrid_astar_failure_reason"),
                "synthetic_hybrid_astar_failure_reason": synth_row.get("hybrid_astar_failure_reason"),
                "base_pose_path_hash": base_row.get("hybrid_astar_pose_path_hash"),
                "synthetic_pose_path_hash": synth_row.get("hybrid_astar_pose_path_hash"),
                "hybrid_path_material": material,
                "default_astar_replaced": False,
                "hybrid_astar_ackermann_feasible_claimed": bool(
                    base_row.get("hybrid_astar_ackermann_feasible_claimed")
                    or synth_row.get("hybrid_astar_ackermann_feasible_claimed")
                ),
                "synthetic_terrain_hash": source["synthetic_terrain_hash"],
                "path_cost_source": PATH_COST_SOURCE,
            }
        )
    return rows


def _viewpoints_from_blockers(blockers: set[tuple[int, int]], width: int, height: int, *, limit: int) -> list[tuple[tuple[int, int], float]]:
    rows: list[tuple[tuple[int, int], float]] = []
    directions = [((-3, 0), 0.0), ((3, 0), 180.0), ((0, -3), 90.0), ((0, 3), 270.0)]
    for blocker in sorted(blockers):
        for (dx, dy), theta in directions:
            cell = (blocker[0] + dx, blocker[1] + dy)
            if 0 <= cell[0] < width and 0 <= cell[1] < height:
                rows.append((cell, theta))
                if len(rows) >= limit:
                    return rows
    return rows


def _hybrid_candidates_from_hard_obstacles(
    source: dict[str, Any],
    width: int,
    height: int,
    *,
    base_grid: Any,
    augmented_grid: Any,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, obstacle in enumerate(_cells(source.get("synthetic_hard_obstacle_cells"))):
        candidate_pairs: list[tuple[tuple[int, int], tuple[int, int], float]] = []
        for offset in (3, 4, 5, 6):
            candidate_pairs.extend(
                [
                    ((obstacle[0] - offset, obstacle[1]), (obstacle[0] + offset, obstacle[1]), 0.0),
                    ((obstacle[0] + offset, obstacle[1]), (obstacle[0] - offset, obstacle[1]), 180.0),
                    ((obstacle[0], obstacle[1] - offset), (obstacle[0], obstacle[1] + offset), 90.0),
                    ((obstacle[0], obstacle[1] + offset), (obstacle[0], obstacle[1] - offset), 270.0),
                ]
            )
        for start, goal, theta_deg in candidate_pairs:
            if not (_in_bounds(start, width, height) and _in_bounds(goal, width, height)):
                continue
            if not base_grid.is_passable(Cell(start[0], start[1])) or not base_grid.is_passable(Cell(goal[0], goal[1])):
                continue
            if not augmented_grid.is_passable(Cell(start[0], start[1])) or not augmented_grid.is_passable(Cell(goal[0], goal[1])):
                continue
            rows.append(
                {
                    "start_cell": start,
                    "candidate": {
                        "scenario_id": f"stage26-synthetic-{index:04d}",
                        "step_index": 0,
                        "candidate_index": index,
                        "candidate_set_hash": f"stage26-synthetic-candidate-set-{source['synthetic_terrain_hash'][:16]}",
                        "candidate_viewpoint": [goal[0], goal[1], theta_deg],
                        "candidate_theta_deg": theta_deg,
                        "path_cost": float(abs(goal[0] - start[0]) + abs(goal[1] - start[1])),
                    },
                }
            )
            break
        if len(rows) >= limit:
            break
    return rows


def _aggregate_sources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "schema_version": "xunce-stage26-0-synthetic-obstacle-source-aggregate/v1",
        "synthetic_terrain_model_id": SYNTHETIC_TERRAIN_MODEL_ID,
        "source_kind": SYNTHETIC_SOURCE_KIND,
        "source_count": len(rows),
        "synthetic_source_ids": [row["synthetic_source_id"] for row in rows],
        "synthetic_rock_count": 0,
        "synthetic_pit_count": 0,
        "synthetic_hard_obstacle_cell_count": 0,
        "synthetic_los_blocker_cell_count": 0,
        "synthetic_high_risk_cell_count": 0,
        "physical_obstacle_cells_written": False,
        "sources": rows,
    }
    hard: set[tuple[int, int]] = set()
    los: set[tuple[int, int]] = set()
    high_risk: set[tuple[int, int]] = set()
    for row in rows:
        aggregate["synthetic_rock_count"] += int(row.get("synthetic_rock_count", 0) or 0)
        aggregate["synthetic_pit_count"] += int(row.get("synthetic_pit_count", 0) or 0)
        hard |= _cells(row.get("synthetic_hard_obstacle_cells"))
        los |= _cells(row.get("synthetic_los_blocker_cells"))
        high_risk |= _cells(row.get("synthetic_high_risk_cells"))
        aggregate["physical_obstacle_cells_written"] = aggregate["physical_obstacle_cells_written"] or bool(
            row.get("physical_obstacle_cells")
        )
    aggregate["synthetic_hard_obstacle_cells"] = _sorted_cells(hard)
    aggregate["synthetic_los_blocker_cells"] = _sorted_cells(los)
    aggregate["synthetic_high_risk_cells"] = _sorted_cells(high_risk)
    aggregate["synthetic_hard_obstacle_cell_count"] = len(hard)
    aggregate["synthetic_los_blocker_cell_count"] = len(los)
    aggregate["synthetic_high_risk_cell_count"] = len(high_risk)
    canonical = json.dumps({k: v for k, v in aggregate.items() if k != "synthetic_terrain_hash"}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    aggregate["synthetic_terrain_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return aggregate


def _map_audit(sidecar_audits: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    blocked_fractions = [float(row.get("blocked_fraction", 0.0)) for row in sidecar_audits]
    effective_blocked_fractions = [float(row.get("effective_blocked_fraction", 0.0)) for row in sidecar_audits]
    return {
        "schema_version": "xunce-stage26-0-map-augmentation-audit/v1",
        "sidecar_count": len(sidecar_audits),
        "synthetic_terrain_hash": source.get("synthetic_terrain_hash"),
        "source_kind": source.get("source_kind"),
        "physical_obstacle_cells_written": bool(source.get("physical_obstacle_cells_written")),
        "sidecar_audits": sidecar_audits,
        "blocked_fraction_max": max(blocked_fractions) if blocked_fractions else None,
        "effective_blocked_fraction_max": max(effective_blocked_fractions) if effective_blocked_fractions else None,
        "blocked_fraction_passed": all(row.get("blocked_fraction_passed") is True for row in sidecar_audits) if sidecar_audits else False,
        "effective_blocked_fraction_passed": all(row.get("effective_blocked_fraction_passed") is True for row in sidecar_audits) if sidecar_audits else False,
        "connectivity_passed": all(row.get("connectivity_passed") is True for row in sidecar_audits) if sidecar_audits else False,
        "effective_connectivity_passed": all(row.get("effective_connectivity_passed") is True for row in sidecar_audits) if sidecar_audits else False,
        "start_clearance_passed": all(row.get("start_clearance_passed") is True for row in sidecar_audits) if sidecar_audits else False,
        "effective_start_clearance_passed": all(row.get("effective_start_clearance_passed") is True for row in sidecar_audits) if sidecar_audits else False,
    }


def _los_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    material = [row for row in rows if row.get("los_material") is True]
    return {
        "schema_version": "xunce-stage26-0-los-impact-audit/v1",
        "row_count": len(rows),
        "los_material_viewpoint_count": len(material),
        "los_material": bool(material),
        "rows": rows,
    }


def _hybrid_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    material = [row for row in rows if row.get("hybrid_path_material") is True]
    return {
        "schema_version": "xunce-stage26-0-hybrid-path-impact-audit/v1",
        "row_count": len(rows),
        "hybrid_path_cost_material_candidate_count": len(material),
        "hybrid_path_material": bool(material),
        "ackermann_feasible_claimed_count": sum(1 for row in rows if row.get("hybrid_astar_ackermann_feasible_claimed") is True),
        "default_astar_replaced_count": sum(1 for row in rows if row.get("default_astar_replaced") is True),
        "rows": rows,
    }


def _route(
    boundary_reasons: list[str],
    input_reasons: list[str],
    map_audit: dict[str, Any],
    source: dict[str, Any],
    los_audit: dict[str, Any],
    hybrid_audit: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "release/default-policy/executor/canary boundary is open"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "required map sidecar inputs are missing"
    if source.get("physical_obstacle_cells_written"):
        return "failed", ROUTE_SOURCE, "synthetic augmentation attempted to write physical obstacle cells"
    if not source.get("synthetic_terrain_hash"):
        return "failed", ROUTE_HASH, "synthetic terrain hash missing"
    if (
        map_audit.get("blocked_fraction_passed") is not True
        or map_audit.get("connectivity_passed") is not True
        or map_audit.get("start_clearance_passed") is not True
        or map_audit.get("effective_blocked_fraction_passed") is not True
        or map_audit.get("effective_connectivity_passed") is not True
        or map_audit.get("effective_start_clearance_passed") is not True
    ):
        return "failed", ROUTE_CONNECTIVITY, "synthetic/effective terrain constraints failed"
    if hybrid_audit.get("ackermann_feasible_claimed_count", 0) != 0 or hybrid_audit.get("default_astar_replaced_count", 0) != 0:
        return "failed", ROUTE_BOUNDARY, "Hybrid A* boundary metadata is invalid"
    if los_audit.get("los_material") is not True:
        return "failed", ROUTE_LOS_DENSITY, "synthetic LOS blockers did not materially affect endpoint theta visibility"
    if hybrid_audit.get("hybrid_path_material") is not True:
        return "failed", ROUTE_HYBRID_DENSITY, "synthetic hard obstacles did not materially affect Hybrid A* path cost"
    return "passed", ROUTE_STAGE26_1, "synthetic terrain contract materially affects LOS and Hybrid A* path cost"


def _audit_blockers(
    map_audit: dict[str, Any],
    source: dict[str, Any],
    los_audit: dict[str, Any],
    hybrid_audit: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if source.get("physical_obstacle_cells_written"):
        reasons.append("synthetic_physical_obstacle_pollution")
    if not source.get("synthetic_terrain_hash"):
        reasons.append("synthetic_terrain_hash_missing")
    if map_audit.get("blocked_fraction_passed") is not True:
        reasons.append("synthetic_blocked_fraction_exceeded")
    if map_audit.get("connectivity_passed") is not True:
        reasons.append("synthetic_connectivity_failed")
    if map_audit.get("start_clearance_passed") is not True:
        reasons.append("synthetic_start_clearance_failed")
    if map_audit.get("effective_blocked_fraction_passed") is not True:
        reasons.append("effective_blocked_fraction_exceeded")
    if map_audit.get("effective_connectivity_passed") is not True:
        reasons.append("effective_connectivity_failed")
    if map_audit.get("effective_start_clearance_passed") is not True:
        reasons.append("effective_start_clearance_failed")
    if los_audit.get("los_material") is not True:
        reasons.append("synthetic_los_not_material")
    if hybrid_audit.get("hybrid_path_material") is not True:
        reasons.append("synthetic_hybrid_path_not_material")
    if hybrid_audit.get("ackermann_feasible_claimed_count", 0) != 0:
        reasons.append("hybrid_astar_ackermann_feasible_claimed")
    if hybrid_audit.get("default_astar_replaced_count", 0) != 0:
        reasons.append("hybrid_astar_default_astar_replaced")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in BOUNDARY_FIELDS:
        if config.get(field) not in (False, 0, 0.0, None):
            reasons.append(f"config_{field}_enabled")
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("config_canary_traffic_fraction_nonzero")
    return reasons


def _input_rejections(config: dict[str, Any], repo_root: Path) -> tuple[list[str], list[Path]]:
    reasons: list[str] = []
    sidecars = _discover_sidecars(config, repo_root)
    if not sidecars:
        reasons.append("source_sidecar_missing")
    if config.get("synthetic_terrain_model_id") != SYNTHETIC_TERRAIN_MODEL_ID:
        reasons.append("synthetic_terrain_model_id_mismatch")
    return reasons, sidecars


def _discover_sidecars(config: dict[str, Any], repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for raw in config.get("source_sidecar_paths", []) or []:
        path = _resolve_path(Path(str(raw)), repo_root)
        if path.is_file():
            paths.append(path)
    if paths:
        return paths[: int(config.get("max_sidecar_count", 2))]
    root_value = config.get("source_roi_expansion_root")
    if root_value:
        root = _resolve_path(Path(str(root_value)), repo_root)
        if root.is_dir():
            paths.extend(sorted(root.rglob("*path-planner-sidecar.json")))
    return paths[: int(config.get("max_sidecar_count", 2))]


def _sidecar_dimensions(sidecar: dict[str, Any]) -> tuple[int, int]:
    cost = sidecar.get("cost")
    if not isinstance(cost, list) or not cost or not isinstance(cost[0], list):
        raise ValueError("sidecar cost must be a non-empty 2D list")
    return len(cost[0]), len(cost)


def _sidecar_resolution_m(sidecar: dict[str, Any]) -> float:
    metadata = sidecar.get("metadata") if isinstance(sidecar.get("metadata"), dict) else {}
    map_source = metadata.get("map_source") if isinstance(metadata.get("map_source"), dict) else {}
    return float(
        sidecar.get("resolution_m")
        or sidecar.get("resolution")
        or metadata.get("resolution_m")
        or metadata.get("resolution")
        or map_source.get("resolution_m")
        or 1.0
    )


def _sidecar_start_cell(sidecar: dict[str, Any], config: dict[str, Any]) -> tuple[int, int]:
    metadata = sidecar.get("metadata") if isinstance(sidecar.get("metadata"), dict) else {}
    for value in (sidecar.get("start_cell"), metadata.get("start_cell"), config.get("start_cell")):
        if isinstance(value, list) and len(value) >= 2:
            return (int(value[0]), int(value[1]))
    return (0, 0)


def _existing_hard_cells(sidecar: dict[str, Any]) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for key in ("physical_obstacle_cells", "obstacle_cells", "slope_blocked_cells", "blocked_cells"):
        result |= _cells(sidecar.get(key))
    result |= _passable_mask_blocked_cells(sidecar.get("passable_mask"))
    return result


def _existing_los_cells(sidecar: dict[str, Any]) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for key in ("physical_obstacle_cells", "obstacle_cells", "slope_blocked_cells"):
        result |= _cells(sidecar.get(key))
    return result


def _passable_mask_blocked_cells(mask: Any) -> set[tuple[int, int]]:
    if not isinstance(mask, list):
        return set()
    result: set[tuple[int, int]] = set()
    for y, row in enumerate(mask):
        if not isinstance(row, list):
            continue
        for x, value in enumerate(row):
            if value is False:
                result.add((x, y))
    return result


def _cells(value: Any) -> set[tuple[int, int]]:
    if not isinstance(value, list):
        return set()
    result: set[tuple[int, int]] = set()
    for cell in value:
        if isinstance(cell, list) and len(cell) >= 2:
            result.add((int(cell[0]), int(cell[1])))
    return result


def _sorted_cells(cells: set[tuple[int, int]]) -> list[list[int]]:
    return [[x, y] for x, y in sorted(cells)]


def _in_bounds(cell: tuple[int, int], width: int, height: int) -> bool:
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = _resolve_path(path, repo_root)
    payload = _read_json(resolved)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    payload.setdefault("synthetic_terrain_model_id", SYNTHETIC_TERRAIN_MODEL_ID)
    payload.setdefault("synthetic_terrain_seed", 260001)
    payload.setdefault("max_traversable_slope_deg", 30.0)
    payload.setdefault("sensor_range_cells", 5)
    payload.setdefault("sensor_fov_deg", 90.0)
    payload.setdefault("max_sidecar_count", 2)
    payload.setdefault("stage26_0_authorized", False)
    payload.setdefault("runs_new_ppo_update", False)
    payload.setdefault("publishes_checkpoint", False)
    payload.setdefault("replaces_default_policy", False)
    payload.setdefault("connects_real_executor", False)
    payload.setdefault("starts_online_canary", False)
    payload.setdefault("canary_traffic_fraction", 0.0)
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.0 Synthetic Rock/Pit Terrain Augmentation Contract",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- synthetic_terrain_hash: `{summary.get('synthetic_terrain_hash')}`",
            f"- synthetic_rock_count: `{summary['synthetic_rock_count']}`",
            f"- synthetic_pit_count: `{summary['synthetic_pit_count']}`",
            f"- synthetic_hard_obstacle_cell_count: `{summary['synthetic_hard_obstacle_cell_count']}`",
            f"- synthetic_los_blocker_cell_count: `{summary['synthetic_los_blocker_cell_count']}`",
            f"- los_material_viewpoint_count: `{summary['los_material_viewpoint_count']}`",
            f"- hybrid_path_cost_material_candidate_count: `{summary['hybrid_path_cost_material_candidate_count']}`",
            "",
            "Stage26.0 adds synthetic terrain proxy evidence only. It does not write physical obstacle truth, run PPO, publish checkpoints, replace default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
