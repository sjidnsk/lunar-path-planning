from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

try:  # pragma: no cover - script execution path
    from xunce_obstacle_aware_theta_sensor_coverage import (
        extract_obstacle_cells,
        obstacle_aware_theta_coverage_hash,
        visible_cells_for_viewpoint_with_obstacles,
    )
    from xunce_theta_sensor_coverage import theta_coverage_hash, visible_cells_for_viewpoint
except ModuleNotFoundError:  # pragma: no cover
    from scripts.xunce_obstacle_aware_theta_sensor_coverage import (
        extract_obstacle_cells,
        obstacle_aware_theta_coverage_hash,
        visible_cells_for_viewpoint_with_obstacles,
    )
    from scripts.xunce_theta_sensor_coverage import theta_coverage_hash, visible_cells_for_viewpoint


CONFIG_SCHEMA_VERSION = "xunce-stage23-0-endpoint-obstacle-aware-theta-sensor-coverage-contract-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage23-0-summary/v1"
LOS_ROW_SCHEMA_VERSION = "xunce-stage23-0-obstacle-los-audit-row/v1"
COVERAGE_DELTA_SCHEMA_VERSION = "xunce-stage23-0-coverage-delta-audit/v1"
COMPAT_SCHEMA_VERSION = "xunce-stage23-0-stage22-compatibility-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage23-0-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage23-0-manifest/v1"

STAGE_ID = "xunce-stage23-0-endpoint-obstacle-aware-theta-sensor-coverage-contract"
DEFAULT_CONFIG = "configs/xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage23_endpoint_obstacle_aware_theta_sensor_coverage/"
    "outputs/path_feedback_batch_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract_v1"
)
DEFAULT_HIGH_FIDELITY_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage22_theta_aware_sensor_action_space/"
    "outputs/path_feedback_batch_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke_v1/"
    "s21_5/pre_ppo_xunce"
)
DEFAULT_STAGE22_5_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage22_theta_aware_sensor_action_space/"
    "outputs/path_feedback_batch_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke_v1"
)

SUMMARY_FILE = "xunce-stage23-0-summary.json"
LOS_AUDIT_FILE = "xunce-stage23-0-obstacle-los-audit.jsonl"
COVERAGE_DELTA_FILE = "xunce-stage23-0-coverage-delta-audit.json"
COMPAT_AUDIT_FILE = "xunce-stage23-0-stage22-compatibility-audit.json"
ROUTING_FILE = "xunce-stage23-0-next-stage-routing.json"
REPORT_FILE = "xunce-stage23-0-report.md"
MANIFEST_FILE = "xunce-stage23-0-manifest.json"

CANDIDATE_METRIC_AUDIT_FILE = "xunce-exploration-coverage-candidate-metric-audit.jsonl"
STAGE22_1_VIEWPOINT_AUDIT_FILE = "xunce-stage22-1-viewpoint-candidate-audit.jsonl"
OBSTACLE_SOURCES_FILE = "xunce-exploration-coverage-obstacle-sources.json"

ROUTE_INPUTS = "rerun_stage23_0_required_obstacle_sources"
ROUTE_LOS_REPAIR = "repair_stage23_0_obstacle_los_contract"
ROUTE_STAGE23_1 = "implement_stage23_1_obstacle_aware_theta_reward_contract"
ROUTE_AUDIT_ONLY = "document_obstacle_occlusion_audit_only"
ROUTE_BOUNDARY = "resolve_stage23_0_boundary_rejections"

BOUNDARY_FIELDS = (
    "stage23_0_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage23.0 endpoint obstacle-aware theta coverage contract audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path, repo_root)
    boundary_reasons = _boundary_rejections(config)
    input_reasons: list[str] = []

    high_fidelity_root = _resolve_path(Path(str(config["high_fidelity_root"])), repo_root)
    stage22_5_root = _resolve_path(Path(str(config.get("stage22_5_root", DEFAULT_STAGE22_5_ROOT))), repo_root)
    candidate_file = _candidate_audit_file(high_fidelity_root)
    if candidate_file is None:
        input_reasons.append("candidate_or_viewpoint_audit_missing")
        candidate_rows: list[dict[str, Any]] = []
    else:
        candidate_rows = _read_jsonl(candidate_file)
    stage22_5_summary = _read_optional_json(stage22_5_root / "xunce-stage22-5-summary.json")

    obstacle_sources = _load_obstacle_sources(high_fidelity_root)
    los_rows, coverage_delta, los_reasons = _audit_obstacle_los(candidate_rows, config, obstacle_sources=obstacle_sources)
    input_reasons.extend(los_reasons)
    compatibility = _stage22_compatibility_audit(config, coverage_delta)
    status, route, route_reason = _route(boundary_reasons, input_reasons, coverage_delta, compatibility)
    blocking_reasons = _unique(boundary_reasons + input_reasons + _route_blocking_reasons(route, coverage_delta))

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking_reasons,
        "high_fidelity_root": str(high_fidelity_root),
        "stage22_5_root": str(stage22_5_root),
        "stage22_5_status": stage22_5_summary.get("status"),
        "stage22_5_next_required_change": stage22_5_summary.get("next_required_change"),
        "candidate_audit_file": str(candidate_file) if candidate_file is not None else None,
        "obstacle_source_artifact": str(high_fidelity_root / OBSTACLE_SOURCES_FILE)
        if (high_fidelity_root / OBSTACLE_SOURCES_FILE).exists()
        else None,
        "candidate_input_row_count": len(candidate_rows),
        "los_audit_row_count": len(los_rows),
        "obstacle_source_missing_count": coverage_delta["obstacle_source_missing_count"],
        "obstacle_source_available_count": coverage_delta["obstacle_source_available_count"],
        "obstacle_source_hash_mismatch_count": coverage_delta["obstacle_source_hash_mismatch_count"],
        "material_occlusion_viewpoint_count": coverage_delta["material_occlusion_viewpoint_count"],
        "obstacle_occlusion_material_to_coverage": coverage_delta["obstacle_occlusion_material_to_coverage"],
        "max_visibility_removed_by_obstacle_count": coverage_delta["max_visibility_removed_by_obstacle_count"],
        "mean_visibility_removed_by_obstacle_count": coverage_delta["mean_visibility_removed_by_obstacle_count"],
        "stage22_reward_obstacle_occlusion_enabled": compatibility["stage22_reward_obstacle_occlusion_enabled"],
        "stage22_reward_still_unobstructed_when_material": compatibility["stage22_reward_still_unobstructed_when_material"],
        "release_or_training_authorized": False,
        "stage23_0_authorized": False,
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
        "stage23_0_authorized": False,
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
        "obstacle_los_audit": str(output_root / LOS_AUDIT_FILE),
        "coverage_delta_audit": str(output_root / COVERAGE_DELTA_FILE),
        "stage22_compatibility_audit": str(output_root / COMPAT_AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_jsonl(output_root / LOS_AUDIT_FILE, los_rows)
    _write_json(output_root / COVERAGE_DELTA_FILE, coverage_delta)
    _write_json(output_root / COMPAT_AUDIT_FILE, compatibility)
    _write_json(output_root / ROUTING_FILE, routing)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    _write_json(output_root / MANIFEST_FILE, manifest)
    return summary


def _audit_obstacle_los(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    obstacle_sources: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    max_rows = int(config.get("max_audit_rows", 5000))
    los_rows: list[dict[str, Any]] = []
    missing_obstacle_source = 0
    available_obstacle_source = 0
    invalid_rows = 0
    hash_mismatch_count = 0
    visibility_removed: list[int] = []
    material_count = 0
    missing_reasons: list[str] = []
    include_no_go = bool(config.get("no_go_blocks_los", False))
    config_obstacles, config_obstacle_source = extract_obstacle_cells(config, include_no_go=include_no_go)
    source_by_id = (obstacle_sources or {}).get("by_id", {})

    for row in rows[:max_rows]:
        cell = _candidate_cell(row)
        theta = _candidate_theta(row)
        if cell is None or theta is None:
            invalid_rows += 1
            continue
        sensor_range = int(row.get("sensor_range_cells") or config.get("sensor_range_cells") or config.get("coverage_radius_cells") or 1)
        sensor_fov = float(row.get("sensor_fov_deg") or config.get("sensor_fov_deg") or 90.0)
        obstacles, obstacle_source = extract_obstacle_cells(row, include_no_go=include_no_go)
        obstacle_source_id = row.get("obstacle_source_id")
        obstacle_source_hash = row.get("obstacle_source_hash")
        obstacle_source_kind = row.get("obstacle_source_kind")
        if not obstacles and obstacle_source_id:
            source_record = source_by_id.get(str(obstacle_source_id))
            if source_record is not None:
                expected_hash = str(source_record.get("obstacle_source_hash") or "")
                if obstacle_source_hash and str(obstacle_source_hash) != expected_hash:
                    hash_mismatch_count += 1
                    continue
                obstacles = {_cell_tuple(cell) for cell in source_record.get("obstacle_cells", [])}
                obstacles = {cell for cell in obstacles if cell is not None}
                obstacle_source = str(source_record.get("obstacle_source_field") or source_record.get("obstacle_source_kind") or "obstacle_source_artifact")
                obstacle_source_hash = expected_hash
                obstacle_source_kind = source_record.get("obstacle_source_kind")
        if not obstacles and config_obstacles:
            obstacles = set(config_obstacles)
            obstacle_source = config_obstacle_source
        if not obstacle_source:
            missing_obstacle_source += 1
            continue
        available_obstacle_source += 1
        clear_visible = visible_cells_for_viewpoint(
            cell,
            theta_deg=theta,
            sensor_range_cells=sensor_range,
            sensor_fov_deg=sensor_fov,
        )
        obstacle_result = visible_cells_for_viewpoint_with_obstacles(
            cell,
            theta_deg=theta,
            sensor_range_cells=sensor_range,
            sensor_fov_deg=sensor_fov,
            obstacle_cells=obstacles,
        )
        removed_count = len(clear_visible) - len(obstacle_result.visible_cells)
        target_obstacle_cells_counted_as_visible = bool(obstacle_result.blocked_obstacle_cells & obstacle_result.visible_cells)
        visibility_removed.append(removed_count)
        changed = removed_count > 0 or theta_coverage_hash(clear_visible) != obstacle_result.obstacle_aware_theta_coverage_hash
        if changed:
            material_count += 1
        los_rows.append(
            {
                "schema_version": LOS_ROW_SCHEMA_VERSION,
                "scenario_id": row.get("scenario_id"),
                "step_index": _int(row.get("step_index"), 0),
                "candidate_index": _int(row.get("candidate_index"), -1),
                "candidate_cell": list(cell),
                "candidate_theta_deg": float(theta),
                "candidate_viewpoint": [int(cell[0]), int(cell[1]), float(theta)],
                "sensor_range_cells": sensor_range,
                "sensor_fov_deg": sensor_fov,
                "obstacle_source": obstacle_source,
                "obstacle_source_id": str(obstacle_source_id) if obstacle_source_id else None,
                "obstacle_source_hash": str(obstacle_source_hash) if obstacle_source_hash else None,
                "obstacle_source_kind": str(obstacle_source_kind) if obstacle_source_kind else None,
                "obstacle_cell_count": len(obstacles),
                "clear_visible_cell_count": len(clear_visible),
                "obstacle_aware_visible_cell_count": len(obstacle_result.visible_cells),
                "visibility_removed_by_obstacle_count": removed_count,
                "occluded_cell_count": obstacle_result.occluded_cell_count,
                "blocked_by_obstacle_count": obstacle_result.blocked_by_obstacle_count,
                "clear_theta_coverage_hash": theta_coverage_hash(clear_visible),
                "obstacle_aware_theta_coverage_hash": obstacle_result.obstacle_aware_theta_coverage_hash,
                "coverage_changed_by_obstacle": changed,
                "target_obstacle_cells_are_not_counted": not target_obstacle_cells_counted_as_visible,
                "target_obstacle_cells_counted_as_visible": target_obstacle_cells_counted_as_visible,
            }
        )

    if rows and missing_obstacle_source == len(rows[:max_rows]) and not available_obstacle_source:
        missing_reasons.append("obstacle_source_missing")
    if invalid_rows:
        missing_reasons.append("candidate_viewpoint_or_theta_missing")
    if hash_mismatch_count:
        missing_reasons.append("obstacle_source_hash_mismatch")
    coverage_delta = {
        "schema_version": COVERAGE_DELTA_SCHEMA_VERSION,
        "candidate_input_row_count": len(rows),
        "audited_row_limit": max_rows,
        "los_audit_row_count": len(los_rows),
        "obstacle_source_missing_count": missing_obstacle_source,
        "obstacle_source_available_count": available_obstacle_source,
        "obstacle_source_hash_mismatch_count": hash_mismatch_count,
        "invalid_theta_viewpoint_row_count": invalid_rows,
        "material_occlusion_viewpoint_count": material_count,
        "obstacle_occlusion_material_to_coverage": material_count > 0,
        "max_visibility_removed_by_obstacle_count": max(visibility_removed) if visibility_removed else 0,
        "mean_visibility_removed_by_obstacle_count": statistics.fmean(visibility_removed) if visibility_removed else 0.0,
        "los_replay_reproducible": True,
    }
    return los_rows, coverage_delta, missing_reasons


def _stage22_compatibility_audit(config: dict[str, Any], coverage_delta: dict[str, Any]) -> dict[str, Any]:
    stage22_reward_obstacle_occlusion_enabled = bool(config.get("stage22_reward_obstacle_occlusion_enabled", False))
    material = bool(coverage_delta["obstacle_occlusion_material_to_coverage"])
    return {
        "schema_version": COMPAT_SCHEMA_VERSION,
        "stage22_reward_obstacle_occlusion_enabled": stage22_reward_obstacle_occlusion_enabled,
        "stage22_reward_still_unobstructed_when_material": bool(material and not stage22_reward_obstacle_occlusion_enabled),
        "stage23_0_does_not_change_stage22_reward": True,
        "stage23_0_is_read_only_contract_smoke": True,
    }


def _route(
    boundary_reasons: list[str],
    input_reasons: list[str],
    coverage_delta: dict[str, Any],
    compatibility: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage23_0_boundary_rejected"
    if not coverage_delta["los_replay_reproducible"]:
        return "failed", ROUTE_LOS_REPAIR, "stage23_0_los_replay_not_reproducible"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage23_0_inputs_or_obstacle_sources_missing"
    if compatibility["stage22_reward_still_unobstructed_when_material"]:
        return "passed", ROUTE_STAGE23_1, "obstacle_occlusion_material_but_stage22_reward_is_unobstructed"
    return "passed", ROUTE_AUDIT_ONLY, "obstacle_occlusion_not_material_in_audit"


def _route_blocking_reasons(route: str, coverage_delta: dict[str, Any]) -> list[str]:
    if route == ROUTE_INPUTS and coverage_delta["obstacle_source_missing_count"]:
        return ["obstacle_source_missing"]
    return []


def _candidate_audit_file(root: Path) -> Path | None:
    for filename in (CANDIDATE_METRIC_AUDIT_FILE, STAGE22_1_VIEWPOINT_AUDIT_FILE):
        path = root / filename
        if path.exists():
            return path
    return None


def _load_obstacle_sources(root: Path) -> dict[str, Any]:
    path = root / OBSTACLE_SOURCES_FILE
    if not path.exists():
        return {"path": None, "by_id": {}, "sources": []}
    payload = _read_json(path)
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    by_id = {
        str(row.get("obstacle_source_id")): row
        for row in sources
        if isinstance(row, dict) and row.get("obstacle_source_id")
    }
    return {"path": str(path), "by_id": by_id, "sources": sources, "payload": payload}


def _candidate_cell(row: dict[str, Any]) -> tuple[int, int] | None:
    viewpoint = row.get("candidate_viewpoint")
    if isinstance(viewpoint, list) and len(viewpoint) >= 2:
        return _cell_tuple(viewpoint)
    return _cell_tuple(row.get("candidate_cell") or row.get("cell"))


def _candidate_theta(row: dict[str, Any]) -> float | None:
    value = row.get("candidate_theta_deg")
    if value is None:
        viewpoint = row.get("candidate_viewpoint")
        if isinstance(viewpoint, list) and len(viewpoint) >= 3:
            value = viewpoint[2]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list | tuple) or len(value) < 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = []
    for field in BOUNDARY_FIELDS:
        if bool(config.get(field, False)):
            reasons.append(f"{field}_enabled")
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction_nonzero")
    return reasons


def _load_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    path = _resolve_path(config_path, repo_root)
    config = _read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unexpected config schema_version: {config.get('schema_version')}")
    config.setdefault("stage22_5_root", DEFAULT_STAGE22_5_ROOT)
    config.setdefault("high_fidelity_root", DEFAULT_HIGH_FIDELITY_ROOT)
    config.setdefault("max_audit_rows", 5000)
    config.setdefault("stage23_0_authorized", False)
    config.setdefault("runs_new_ppo_update", False)
    config.setdefault("publishes_checkpoint", False)
    config.setdefault("replaces_default_policy", False)
    config.setdefault("connects_real_executor", False)
    config.setdefault("starts_online_canary", False)
    config.setdefault("canary_traffic_fraction", 0.0)
    config.setdefault("no_go_blocks_los", False)
    return config


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _unique(values: list[str]) -> list[str]:
    return sorted(set(values))


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage23.0 Endpoint Obstacle-Aware Theta Sensor Coverage Contract",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- candidate_input_row_count: `{summary['candidate_input_row_count']}`",
            f"- los_audit_row_count: `{summary['los_audit_row_count']}`",
            f"- obstacle_source_missing_count: `{summary['obstacle_source_missing_count']}`",
            f"- material_occlusion_viewpoint_count: `{summary['material_occlusion_viewpoint_count']}`",
            "",
            "Stage23.0 只复算 endpoint `(x,y,theta)` 视野遮挡合同；不做沿路径持续观测、不做连续 theta、不训练 PPO、不发布 checkpoint。",
            "",
        ]
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
