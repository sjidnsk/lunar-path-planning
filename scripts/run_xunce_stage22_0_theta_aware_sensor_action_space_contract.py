from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by script execution
    from xunce_theta_sensor_coverage import (
        candidate_viewpoint_hash,
        expand_candidate_viewpoints,
        theta_bins,
        theta_coverage_hash,
        visible_cells_for_viewpoint,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.xunce_theta_sensor_coverage import (
        candidate_viewpoint_hash,
        expand_candidate_viewpoints,
        theta_bins,
        theta_coverage_hash,
        visible_cells_for_viewpoint,
    )


CONFIG_SCHEMA_VERSION = "xunce-stage22-0-theta-aware-sensor-action-space-contract-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage22-0-summary/v1"
ACTION_CONTRACT_SCHEMA_VERSION = "xunce-stage22-0-theta-action-contract-audit/v1"
FOOTPRINT_ROW_SCHEMA_VERSION = "xunce-stage22-0-sensor-footprint-audit-row/v1"
EXPANSION_SCHEMA_VERSION = "xunce-stage22-0-action-space-expansion-audit/v1"
PPO_IMPACT_SCHEMA_VERSION = "xunce-stage22-0-ppo-contract-impact/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage22-0-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage22-0-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage22_0_theta_aware_sensor_action_space_contract_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage22_theta_aware_sensor_action_space/"
    "outputs/path_feedback_batch_xunce_stage22_0_theta_aware_sensor_action_space_contract_v1"
)

SUMMARY_FILE = "xunce-stage22-0-summary.json"
ACTION_CONTRACT_FILE = "xunce-stage22-0-theta-action-contract-audit.json"
FOOTPRINT_AUDIT_FILE = "xunce-stage22-0-sensor-footprint-audit.jsonl"
EXPANSION_FILE = "xunce-stage22-0-action-space-expansion-audit.json"
PPO_IMPACT_FILE = "xunce-stage22-0-ppo-contract-impact.json"
ROUTING_FILE = "xunce-stage22-0-next-stage-routing.json"
REPORT_FILE = "xunce-stage22-0-report.md"
MANIFEST_FILE = "xunce-stage22-0-manifest.json"

ROUTE_INPUTS = "rerun_stage22_0_required_inputs"
ROUTE_THETA_OPTIONAL = "document_theta_optional_sensor_model"
ROUTE_STAGE22_1 = "implement_stage22_1_theta_aware_candidate_viewpoint_generation"
ROUTE_STAGE22_2 = "run_stage22_2_theta_aware_coverage_reward_contract"
ROUTE_BOUNDARY = "resolve_stage22_0_boundary_rejections"

BOUNDARY_FIELDS = (
    "stage22_0_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage22.0 theta-aware sensor/action-space contract audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage22_0_theta_aware_sensor_action_space_contract(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    return 1 if summary["status"] == "failed" else 0


def run_xunce_stage22_0_theta_aware_sensor_action_space_contract(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path, repo_root)
    boundary_reasons = _boundary_rejections(config)
    roots, input_reasons = _discover_stage21_roots(config)
    rows, row_reasons = _load_stage21_rows(roots)
    input_reasons.extend(row_reasons)
    if not rows:
        input_reasons.append("stage21_trainable_rows_missing")

    theta_values = theta_bins(theta_bin_count=int(config["theta_bin_count"]), theta_step_deg=int(config["theta_step_deg"]))
    footprint_rows, action_contract, expansion, ppo_impact = _audit_theta_contract(rows, config, theta_values)
    status, route, route_reason = _route(input_reasons, boundary_reasons, action_contract, ppo_impact)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": "xunce-stage22-0-theta-aware-sensor-action-space-contract",
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "input_reason_codes": sorted(set(input_reasons)),
        "boundary_reason_codes": sorted(set(boundary_reasons)),
        "stage21_19_root": roots.get("stage21_19_root"),
        "stage21_19_status": roots.get("stage21_19_status"),
        "stage21_19_next_required_change": roots.get("stage21_19_next_required_change"),
        "stage21_1_root_count": len(roots["stage21_1_roots"]),
        "stage21_3_root_count": len(roots["stage21_3_roots"]),
        "audited_transition_count": len(rows),
        "theta_bin_count": int(config["theta_bin_count"]),
        "theta_step_deg": int(config["theta_step_deg"]),
        "theta_values": theta_values,
        "sensor_model_id": config["sensor_model_id"],
        "sensor_fov_deg": float(config["sensor_fov_deg"]),
        "sensor_range_cells": int(config["sensor_range_cells"]),
        "occlusion_audit_only": bool(config["occlusion_audit_only"]),
        "point_only_action_space_detected": action_contract["point_only_action_space_detected"],
        "theta_material_to_coverage": action_contract["theta_material_to_coverage"],
        "theta_changes_action_preference": action_contract["theta_changes_action_preference"],
        "stage21_ppo_batch_theta_incompatible": ppo_impact["stage21_ppo_batch_theta_incompatible"],
        "old_point_only_evidence_not_theta_readiness": ppo_impact["old_point_only_evidence_not_theta_readiness"],
        "sensor_footprint_audit_row_count": len(footprint_rows),
        "sensor_footprint_audit_truncated": action_contract["sensor_footprint_audit_truncated"],
        "release_or_training_authorized": False,
        "stage22_0_authorized": False,
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
        "stage22_0_authorized": False,
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
        "theta_action_contract_audit": str(output_root / ACTION_CONTRACT_FILE),
        "sensor_footprint_audit": str(output_root / FOOTPRINT_AUDIT_FILE),
        "action_space_expansion_audit": str(output_root / EXPANSION_FILE),
        "ppo_contract_impact": str(output_root / PPO_IMPACT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / ACTION_CONTRACT_FILE, action_contract)
    _write_jsonl(output_root / FOOTPRINT_AUDIT_FILE, footprint_rows)
    _write_json(output_root / EXPANSION_FILE, expansion)
    _write_json(output_root / PPO_IMPACT_FILE, ppo_impact)
    _write_json(output_root / ROUTING_FILE, routing)
    (output_root / REPORT_FILE).write_text(_render_report(summary, action_contract, ppo_impact), encoding="utf-8")
    _write_json(output_root / MANIFEST_FILE, manifest)
    return summary


def _audit_theta_contract(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    theta_values: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    max_rows = int(config["max_sensor_footprint_audit_rows"])
    threshold = int(config["material_new_cell_delta_threshold"])
    sensor_model_id = str(config["sensor_model_id"])
    sensor_range = int(config["sensor_range_cells"])
    sensor_fov = float(config["sensor_fov_deg"])
    covered_by_source_scenario: dict[tuple[str, str], set[tuple[int, int]]] = {}

    footprint_rows: list[dict[str, Any]] = []
    point_only_rows = 0
    theta_contract_rows = 0
    candidate_count_total = 0
    material_step_count = 0
    preference_change_count = 0
    material_candidate_count = 0
    old_best_join_count = 0
    theta_best_join_count = 0
    max_old_action_count = 0
    max_expanded_action_count = 0

    for row in rows:
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        observation = row.get("observation") if isinstance(row.get("observation"), dict) else {}
        scenario_id = str(row.get("scenario_id") or info.get("scenario_id") or "unknown-scenario")
        current_cell = _cell_tuple(info.get("current_cell_before") or row.get("current_cell") or observation.get("current_cell"))
        candidate_cells = _candidate_cells(row)
        if current_cell is None or not candidate_cells:
            continue
        action_mask = _bool_list(info.get("sampling_mask") or info.get("action_mask") or observation.get("action_mask") or [True] * len(candidate_cells))
        valid_candidates = [
            (index, cell)
            for index, cell in enumerate(candidate_cells)
            if index >= len(action_mask) or action_mask[index]
        ]
        candidate_count_total += len(valid_candidates)
        max_old_action_count = max(max_old_action_count, len(valid_candidates))
        max_expanded_action_count = max(max_expanded_action_count, len(valid_candidates) * len(theta_values))
        expanded = expand_candidate_viewpoints(
            [list(cell) for _, cell in valid_candidates],
            theta_values=theta_values,
            sensor_model_id=sensor_model_id,
            sensor_range_cells=sensor_range,
            sensor_fov_deg=sensor_fov,
        )
        if _row_has_theta_contract(row):
            theta_contract_rows += 1
        else:
            point_only_rows += 1
        source_key = (str(row.get("_stage22_source_root") or ""), scenario_id)
        covered = covered_by_source_scenario.setdefault(source_key, set())
        step_material = False
        candidate_best_new_counts: list[float] = [float("-inf") for _ in candidate_cells]
        selected_cell = _cell_tuple(info.get("selected_cell"))
        selected_best_visible: set[tuple[int, int]] = set()

        for original_index, cell in valid_candidates:
            per_theta: list[tuple[int, set[tuple[int, int]], int]] = []
            for theta in theta_values:
                visible = visible_cells_for_viewpoint(
                    cell,
                    theta_deg=theta,
                    sensor_range_cells=sensor_range,
                    sensor_fov_deg=sensor_fov,
                )
                new_count = len(visible - covered)
                per_theta.append((theta, visible, new_count))
                if len(footprint_rows) < max_rows:
                    cost = _candidate_path_cost(row, original_index)
                    footprint_rows.append(
                        {
                            "schema_version": FOOTPRINT_ROW_SCHEMA_VERSION,
                            "scenario_id": scenario_id,
                            "step_index": _int(row.get("step_index") or info.get("step_index"), 0),
                            "current_cell": list(current_cell),
                            "candidate_index": int(original_index),
                            "candidate_cell": list(cell),
                            "candidate_theta_deg": int(theta),
                            "candidate_viewpoint": [int(cell[0]), int(cell[1]), int(theta)],
                            "sensor_model_id": sensor_model_id,
                            "sensor_fov_deg": sensor_fov,
                            "sensor_range_cells": sensor_range,
                            "theta_visible_cell_count": len(visible),
                            "theta_new_visible_cell_count": new_count,
                            "theta_coverage_hash": theta_coverage_hash(visible),
                            "theta_coverage_gain_per_path_cost": _safe_ratio(float(new_count), cost),
                            "candidate_set_hash": info.get("candidate_set_hash") or row.get("candidate_set_hash"),
                            "theta_candidate_set_hash": candidate_viewpoint_hash(expanded),
                            "occlusion_audit_only": bool(config["occlusion_audit_only"]),
                        }
                    )
            counts = [item[2] for item in per_theta]
            if counts and max(counts) - min(counts) >= threshold:
                step_material = True
                material_candidate_count += 1
            best_theta, best_visible, best_new_count = max(per_theta, key=lambda item: (item[2], -abs(item[0])))
            candidate_best_new_counts[original_index] = float(best_new_count)
            if selected_cell == cell:
                selected_best_visible = set(best_visible)

        old_best_index = _old_best_candidate_index(row)
        if old_best_index is not None:
            old_best_join_count += 1
        theta_best_index = _argmax(candidate_best_new_counts)
        if theta_best_index is not None:
            theta_best_join_count += 1
        if old_best_index is not None and theta_best_index is not None and old_best_index != theta_best_index:
            preference_change_count += 1
        if step_material:
            material_step_count += 1
        if selected_best_visible:
            covered.update(selected_best_visible)

    row_count = len(rows)
    point_only = point_only_rows > 0
    theta_material = material_step_count > 0
    preference_changed = preference_change_count > 0
    action_contract = {
        "schema_version": ACTION_CONTRACT_SCHEMA_VERSION,
        "old_action_contract": "candidate_cell=(x,y)",
        "new_action_contract": "candidate_viewpoint=(x,y,theta_deg)",
        "candidate_set_hash_must_include_theta": True,
        "point_only_action_space_detected": point_only,
        "point_only_row_count": point_only_rows,
        "theta_contract_row_count": theta_contract_rows,
        "theta_material_to_coverage": theta_material,
        "theta_material_step_count": material_step_count,
        "theta_material_candidate_count": material_candidate_count,
        "theta_changes_action_preference": preference_changed,
        "theta_preference_change_count": preference_change_count,
        "old_best_candidate_join_count": old_best_join_count,
        "theta_best_candidate_join_count": theta_best_join_count,
        "sensor_footprint_audit_truncated": candidate_count_total * len(theta_values) > max_rows,
    }
    expansion = {
        "schema_version": EXPANSION_SCHEMA_VERSION,
        "audited_transition_count": row_count,
        "old_point_candidate_count_total": candidate_count_total,
        "theta_value_count": len(theta_values),
        "expanded_viewpoint_candidate_count_total": candidate_count_total * len(theta_values),
        "max_old_action_count_per_step": max_old_action_count,
        "max_expanded_action_count_per_step": max_expanded_action_count,
        "action_space_multiplier": len(theta_values),
    }
    ppo_impact = {
        "schema_version": PPO_IMPACT_SCHEMA_VERSION,
        "stage21_ppo_batch_theta_incompatible": point_only,
        "stage21_batch_action_semantics": "candidate_index_over_candidate_cells",
        "stage22_required_action_semantics": "candidate_index_over_candidate_viewpoints",
        "old_point_only_evidence_not_theta_readiness": bool(point_only and theta_material),
        "theta_aware_network_change_required": bool(point_only),
        "theta_aware_collector_change_required": bool(point_only),
        "theta_aware_stage21_batch_contract_change_required": bool(point_only),
    }
    return footprint_rows, action_contract, expansion, ppo_impact


def _route(
    input_reasons: list[str],
    boundary_reasons: list[str],
    action_contract: dict[str, Any],
    ppo_impact: dict[str, Any],
) -> tuple[str, str, str]:
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage22_0_inputs_missing_or_unreadable"
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage22_0_boundary_rejected"
    if not action_contract["theta_material_to_coverage"]:
        return "passed", ROUTE_THETA_OPTIONAL, "theta_did_not_materially_change_coverage_in_audit"
    if ppo_impact["stage21_ppo_batch_theta_incompatible"]:
        return "passed", ROUTE_STAGE22_1, "theta_material_but_stage21_point_only_batch_detected"
    return "passed", ROUTE_STAGE22_2, "theta_aware_contract_already_available"


def _discover_stage21_roots(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    stage21_1_roots = [str(item) for item in config.get("stage21_1_roots", []) if str(item)]
    stage21_3_roots = [str(item) for item in config.get("stage21_3_roots", []) if str(item)]
    configured_stage21_19 = str(config.get("stage21_19_root") or "")
    stage21_19_root = Path(configured_stage21_19) if configured_stage21_19 else None
    stage21_19_status = None
    stage21_19_route = None
    stage21_18_roots: list[Path] = []
    configured_stage21_18 = str(config.get("stage21_18_root") or "")
    if configured_stage21_18:
        stage21_18_roots.append(Path(configured_stage21_18))
    if stage21_19_root is not None and stage21_19_root.exists():
        summary19 = _read_json(stage21_19_root / "xunce-stage21-19-summary.json")
        stage21_19_status = summary19.get("status")
        stage21_19_route = summary19.get("next_required_change")
        if summary19.get("stage21_18_root"):
            stage21_18_roots.append(Path(str(summary19["stage21_18_root"])))
    elif stage21_19_root is not None:
        reasons.append("stage21_19_root_missing")

    existing_stage21_18_roots: list[Path] = []
    missing_stage21_18_count = 0
    for stage21_18_root in stage21_18_roots:
        if stage21_18_root.exists():
            existing_stage21_18_roots.append(stage21_18_root)
        else:
            missing_stage21_18_count += 1
    if missing_stage21_18_count > 0 and not existing_stage21_18_roots:
        reasons.append("stage21_18_root_missing")

    for stage21_18_root in existing_stage21_18_roots:
        round_rows = _read_jsonl(stage21_18_root / "xunce-stage21-18-round-results.jsonl")
        for round_row in round_rows:
            for root_value in round_row.get("stage21_6_roots") or []:
                stage21_6_root = Path(str(root_value))
                for seed_row in _read_jsonl(stage21_6_root / "xunce-stage21-6-seed-results.jsonl"):
                    if seed_row.get("stage21_1_root"):
                        stage21_1_roots.append(str(seed_row["stage21_1_root"]))
                    if seed_row.get("stage21_3_root"):
                        stage21_3_roots.append(str(seed_row["stage21_3_root"]))
    stage21_1_roots = sorted(set(stage21_1_roots))
    stage21_3_roots = sorted(set(stage21_3_roots))
    if not stage21_1_roots and not stage21_3_roots:
        reasons.append("stage21_1_or_stage21_3_roots_missing")
    return {
        "stage21_1_roots": stage21_1_roots,
        "stage21_3_roots": stage21_3_roots,
        "stage21_19_root": str(stage21_19_root) if stage21_19_root is not None else None,
        "stage21_19_status": stage21_19_status,
        "stage21_19_next_required_change": stage21_19_route,
    }, sorted(set(reasons))


def _load_stage21_rows(roots: dict[str, list[str]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    for root in roots["stage21_3_roots"]:
        path = Path(root) / "xunce-stage21-3-ppo-trainable-batch.jsonl"
        batch_rows, reason = _read_jsonl_checked(path, missing_reason="stage21_3_trainable_batch_missing", empty_reason="stage21_3_trainable_batch_empty")
        if reason:
            reasons.append(reason)
        for row in batch_rows:
            row["_stage22_source_type"] = "stage21_3_validated_batch"
            row["_stage22_source_root"] = str(root)
            rows.append(row)
    if rows:
        return rows, sorted(set(reasons))
    for root in roots["stage21_1_roots"]:
        path = Path(root) / "xunce-stage21-1-ppo-trainable-batch.jsonl"
        batch_rows, reason = _read_jsonl_checked(path, missing_reason="stage21_1_trainable_batch_missing", empty_reason="stage21_1_trainable_batch_empty")
        if reason:
            reasons.append(reason)
        for row in batch_rows:
            row["_stage22_source_type"] = "stage21_1_collector_batch"
            row["_stage22_source_root"] = str(root)
            rows.append(row)
    return rows, sorted(set(reasons))


def _candidate_cells(row: dict[str, Any]) -> list[tuple[int, int]]:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    observation = row.get("observation") if isinstance(row.get("observation"), dict) else {}
    raw = info.get("candidate_cells") or observation.get("candidate_cells") or row.get("candidate_cells") or []
    cells: list[tuple[int, int]] = []
    for item in raw:
        cell = _cell_tuple(item)
        if cell is not None:
            cells.append(cell)
    return cells


def _row_has_theta_contract(row: dict[str, Any]) -> bool:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    observation = row.get("observation") if isinstance(row.get("observation"), dict) else {}
    for key in ("candidate_theta_deg", "candidate_viewpoint"):
        if key in row or key in info or key in observation:
            return True
    raw_candidates = info.get("candidate_cells") or observation.get("candidate_cells") or row.get("candidate_cells") or []
    if isinstance(raw_candidates, list) and any(isinstance(item, list) and len(item) >= 3 for item in raw_candidates):
        return True
    observation_candidates = observation.get("candidate_cells") or []
    return isinstance(observation_candidates, list) and any(isinstance(item, list) and len(item) >= 3 for item in observation_candidates)


def _old_best_candidate_index(row: dict[str, Any]) -> int | None:
    observation = row.get("observation") if isinstance(row.get("observation"), dict) else {}
    names = observation.get("candidate_feature_names") or []
    features = observation.get("candidate_features") or []
    if "expected_coverage_rate_delta" in names:
        index = names.index("expected_coverage_rate_delta")
        values: list[float] = []
        for row_features in features:
            try:
                values.append(float(row_features[index]))
            except (TypeError, ValueError, IndexError):
                values.append(float("-inf"))
        return _argmax(values)
    try:
        return int(row.get("action_index"))
    except (TypeError, ValueError):
        return None


def _candidate_path_cost(row: dict[str, Any], candidate_index: int) -> float | None:
    observation = row.get("observation") if isinstance(row.get("observation"), dict) else {}
    names = observation.get("candidate_feature_names") or []
    features = observation.get("candidate_features") or []
    if "path_cost" in names and candidate_index < len(features):
        try:
            return float(features[candidate_index][names.index("path_cost")])
        except (TypeError, ValueError, IndexError):
            return None
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    try:
        return float(info.get("path_cost"))
    except (TypeError, ValueError):
        return None


def _load_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    path = config_path if config_path.is_absolute() else repo_root / config_path
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported config schema_version: {payload.get('schema_version')}")
    defaults = {
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_model_id": "theta-fov-90-range-radius/v1",
        "sensor_fov_deg": 90,
        "sensor_range_cells": 1,
        "occlusion_audit_only": True,
        "material_new_cell_delta_threshold": 1,
        "max_sensor_footprint_audit_rows": 5000,
        "stage21_1_roots": [],
        "stage21_3_roots": [],
        "stage22_0_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    config["theta_bin_count"] = _positive_int(config["theta_bin_count"], "theta_bin_count")
    config["theta_step_deg"] = _positive_int(config["theta_step_deg"], "theta_step_deg")
    config["sensor_range_cells"] = _positive_int(config["sensor_range_cells"], "sensor_range_cells")
    config["sensor_fov_deg"] = _positive_float(config["sensor_fov_deg"], "sensor_fov_deg")
    config["material_new_cell_delta_threshold"] = _positive_int(config["material_new_cell_delta_threshold"], "material_new_cell_delta_threshold")
    config["max_sensor_footprint_audit_rows"] = _positive_int(config["max_sensor_footprint_audit_rows"], "max_sensor_footprint_audit_rows")
    return config


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is not False]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _render_report(summary: dict[str, Any], action_contract: dict[str, Any], ppo_impact: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage22.0 Theta-Aware Sensor Action Space Contract",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- point_only_action_space_detected: `{summary['point_only_action_space_detected']}`",
            f"- theta_material_to_coverage: `{summary['theta_material_to_coverage']}`",
            f"- theta_changes_action_preference: `{summary['theta_changes_action_preference']}`",
            f"- stage21_ppo_batch_theta_incompatible: `{summary['stage21_ppo_batch_theta_incompatible']}`",
            "",
            "Current Stage21 PPO evidence uses candidate point indexes over `(x,y)`. Stage22 requires viewpoint actions `(x,y,theta_deg)` because sensor heading can change the visible cells after reaching the same point.",
            "",
            f"- old action contract: `{action_contract['old_action_contract']}`",
            f"- new action contract: `{action_contract['new_action_contract']}`",
            f"- old point-only evidence blocked for theta readiness: `{ppo_impact['old_point_only_evidence_not_theta_readiness']}`",
            "",
            "Stage22.0 is a read-only audit. It does not train PPO, publish a checkpoint, replace the default policy, connect an executor, or start canary traffic.",
        ]
    ) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_jsonl_checked(path: Path, *, missing_reason: str, empty_reason: str) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], missing_reason
    rows = _read_jsonl(path)
    if not rows:
        return [], empty_reason
    return rows, None


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


def _bool_list(value: Any) -> list[bool]:
    if not isinstance(value, list):
        return []
    return [bool(item) for item in value]


def _argmax(values: list[float | int]) -> int | None:
    if not values:
        return None
    best_index = 0
    best_value = float(values[0])
    for index, value in enumerate(values[1:], start=1):
        if float(value) > best_value:
            best_index = index
            best_value = float(value)
    return best_index


def _safe_ratio(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
