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

from xunce_hybrid_astar_candidate_path_cost import (  # noqa: E402
    PATH_COST_SOURCE,
    build_cost_grid_from_config,
    declared_obstacle_cells,
    evaluate_hybrid_astar_candidate_path_cost,
)


STAGE_ID = "xunce-stage24-1-hybrid-astar-candidate-path-cost-integration"
CONFIG_SCHEMA_VERSION = "xunce-stage24-1-hybrid-astar-candidate-path-cost-integration-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage24-1-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage24-1-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage24-1-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage24_1_hybrid_astar_candidate_path_cost_integration_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage24_hybrid_astar_pose_planner/"
    "outputs/path_feedback_batch_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration_v1"
)

SUMMARY_FILE = "xunce-stage24-1-summary.json"
CANDIDATE_AUDIT_FILE = "xunce-stage24-1-candidate-hybrid-path-cost-audit.jsonl"
DELTA_AUDIT_FILE = "xunce-stage24-1-grid-vs-hybrid-cost-delta-audit.json"
RECOMMENDATION_FILE = "xunce-stage24-1-path-cost-source-recommendation.json"
ROUTING_FILE = "xunce-stage24-1-next-stage-routing.json"
REPORT_FILE = "xunce-stage24-1-report.md"
MANIFEST_FILE = "xunce-stage24-1-manifest.json"

ROUTE_INPUTS = "rerun_stage24_1_required_inputs"
ROUTE_POSE_CONTRACT = "repair_stage24_1_candidate_pose_contract"
ROUTE_PLANNER_CONTRACT = "repair_stage24_1_hybrid_astar_candidate_planner_contract"
ROUTE_STAGE24_2 = "run_stage24_2_hybrid_astar_reward_path_cost_contract"
ROUTE_AUDIT_ONLY = "document_stage24_1_hybrid_astar_audit_only"
ROUTE_BOUNDARY = "resolve_stage24_1_boundary_rejections"

BOUNDARY_FIELDS = (
    "stage24_1_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage24.1 Hybrid A* candidate path cost integration audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    platform_contract, platform_hash = _load_platform_contract(config["platform_contract"], repo_root)
    boundary_reasons = _boundary_rejections(config, platform_contract)
    input_reasons, stage24_0_summary = _input_rejections(config, repo_root, platform_contract, platform_hash)
    rows: list[dict[str, Any]] = []
    if not boundary_reasons and not input_reasons:
        grid = build_cost_grid_from_config(config["grid"])
        rows = _evaluate_rows(config, grid, platform_contract, platform_hash)

    delta_audit = _delta_audit(rows, float(config["material_cost_delta_threshold"]))
    recommendation = _recommendation(config, platform_contract, platform_hash, delta_audit)
    status, route, route_reason = _route(boundary_reasons, input_reasons, rows, delta_audit)
    blocking_reasons = [*boundary_reasons, *input_reasons, *_row_blockers(rows)]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking_reasons,
        "stage24_0_root": config.get("stage24_0_root"),
        "stage24_0_status": stage24_0_summary.get("status") if stage24_0_summary else None,
        "stage24_0_next_required_change": stage24_0_summary.get("next_required_change") if stage24_0_summary else None,
        "platform_contract_id": platform_contract.get("platform_id"),
        "platform_contract_hash": platform_hash,
        "max_traversable_slope_deg": float(platform_contract["terrain_policy"]["max_traversable_slope_deg"]),
        "path_cost_source_recommendation": PATH_COST_SOURCE,
        "hybrid_astar_pose_path_cost_enabled": bool(config["hybrid_astar_pose_path_cost_enabled"]),
        "hybrid_astar_candidate_row_count": len(rows),
        "candidate_pose_contract_invalid_count": sum(1 for row in rows if not row.get("candidate_pose_contract_valid")),
        "hybrid_astar_reachable_count": sum(1 for row in rows if row.get("hybrid_astar_reachable") is True),
        "legacy_grid_astar_reachable_count": sum(1 for row in rows if row.get("legacy_grid_astar_reachable") is True),
        "hybrid_vs_grid_cost_material": bool(delta_audit["hybrid_vs_grid_cost_material"]),
        "material_cost_delta_count": int(delta_audit["material_cost_delta_count"]),
        "default_astar_replaced": False,
        "ackermann_feasible_claimed": any(row.get("hybrid_astar_ackermann_feasible_claimed") is True for row in rows),
        "stage24_1_authorized": False,
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
        "stage24_1_authorized": False,
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
        "candidate_hybrid_path_cost_audit": str(output_root / CANDIDATE_AUDIT_FILE),
        "grid_vs_hybrid_cost_delta_audit": str(output_root / DELTA_AUDIT_FILE),
        "path_cost_source_recommendation": str(output_root / RECOMMENDATION_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_jsonl(output_root / CANDIDATE_AUDIT_FILE, rows)
    _write_json(output_root / DELTA_AUDIT_FILE, delta_audit)
    _write_json(output_root / RECOMMENDATION_FILE, recommendation)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary, delta_audit), encoding="utf-8")
    return summary


def _evaluate_rows(config: dict[str, Any], grid: Any, platform_contract: dict[str, Any], platform_hash: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_pose = config.get("current_pose")
    obstacle_cells = declared_obstacle_cells(config["grid"])
    for candidate in config["candidate_rows"]:
        row = evaluate_hybrid_astar_candidate_path_cost(
            grid=grid,
            current_pose=current_pose,
            candidate=candidate,
            platform_contract_hash=platform_hash,
            max_traversable_slope_deg=float(platform_contract["terrain_policy"]["max_traversable_slope_deg"]),
            theta_bin_count=int(config["theta_bin_count"]),
            goal_theta_tolerance_deg=float(config["goal_theta_tolerance_deg"]),
            max_iterations=int(config["max_iterations"]),
            primitive_duration_s=float(config["primitive_duration_s"]),
            integration_dt_s=float(config["integration_dt_s"]),
            max_speed_mps=float(config["max_speed_mps"]),
            max_angular_speed_degps=float(config["max_angular_speed_degps"]),
            footprint_length_m=float(platform_contract["base"]["body_length_m"]),
            footprint_width_m=float(platform_contract["base"]["body_width_m"]),
            footprint_safety_margin_m=float(config["footprint_safety_margin_m"]),
            rotation_cost_weight=float(config["rotation_cost_weight"]),
            reverse_penalty_weight=float(config["reverse_penalty_weight"]),
            turn_penalty_weight=float(config["turn_penalty_weight"]),
        )
        row.update(
            {
                "current_pose": current_pose,
                "current_pose_provenance": config.get("current_pose_provenance"),
                "grid_resolution_m": config["grid"].get("resolution_m", config["grid"].get("resolution")),
                "slope_obstacle_source_hash": config["grid"].get("slope_obstacle_source_hash"),
                "declared_obstacle_cell_count": len(obstacle_cells),
            }
        )
        rows.append(row)
    return rows


def _delta_audit(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    deltas = [
        float(row["hybrid_vs_grid_path_cost_delta"])
        for row in rows
        if row.get("hybrid_vs_grid_path_cost_delta") is not None and math.isfinite(float(row["hybrid_vs_grid_path_cost_delta"]))
    ]
    material = [delta for delta in deltas if abs(delta) > threshold]
    return {
        "schema_version": "xunce-stage24-1-grid-vs-hybrid-cost-delta-audit/v1",
        "row_count": len(rows),
        "comparable_cost_row_count": len(deltas),
        "material_cost_delta_threshold": float(threshold),
        "material_cost_delta_count": len(material),
        "hybrid_vs_grid_cost_material": bool(material),
        "hybrid_vs_grid_path_cost_delta_min": min(deltas) if deltas else None,
        "hybrid_vs_grid_path_cost_delta_max": max(deltas) if deltas else None,
        "hybrid_vs_grid_path_cost_delta_mean": sum(deltas) / len(deltas) if deltas else None,
    }


def _recommendation(config: dict[str, Any], platform_contract: dict[str, Any], platform_hash: str, delta_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage24-1-path-cost-source-recommendation/v1",
        "recommended_path_cost_source": PATH_COST_SOURCE,
        "recommendation_scope": "audit_only_not_global_default",
        "hybrid_astar_pose_path_cost_enabled": bool(config["hybrid_astar_pose_path_cost_enabled"]),
        "hybrid_vs_grid_cost_material": bool(delta_audit["hybrid_vs_grid_cost_material"]),
        "platform_contract_id": platform_contract.get("platform_id"),
        "platform_contract_hash": platform_hash,
        "max_traversable_slope_deg": float(platform_contract["terrain_policy"]["max_traversable_slope_deg"]),
        "default_astar_replaced": False,
        "ackermann_feasible_claimed": False,
    }


def _route(
    boundary_reasons: list[str],
    input_reasons: list[str],
    rows: list[dict[str, Any]],
    delta_audit: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "release/default-policy/executor/canary boundary is open"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "required Stage24.0/platform/candidate inputs are missing or invalid"
    if any(not row.get("candidate_pose_contract_valid") for row in rows):
        return "failed", ROUTE_POSE_CONTRACT, "candidate/current pose/theta contract failed"
    if any(row.get("hybrid_astar_ackermann_feasible_claimed") is True for row in rows):
        return "failed", ROUTE_PLANNER_CONTRACT, "Hybrid A* incorrectly claimed Ackermann feasibility"
    if not any(row.get("hybrid_astar_reachable") is True for row in rows):
        return "failed", ROUTE_PLANNER_CONTRACT, "Hybrid A* could not solve any candidate pose path"
    if delta_audit["hybrid_vs_grid_cost_material"]:
        return "passed", ROUTE_STAGE24_2, "Hybrid A* pose cost differs materially from grid A* candidate cost"
    return "passed", ROUTE_AUDIT_ONLY, "Hybrid A* pose cost audit passed but was not materially different from grid A*"


def _row_blockers(rows: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if any(not row.get("candidate_pose_contract_valid") for row in rows):
        reasons.append("candidate_pose_contract_invalid")
    if rows and not any(row.get("hybrid_astar_reachable") is True for row in rows):
        reasons.append("hybrid_astar_no_reachable_candidate")
    if any(row.get("hybrid_astar_ackermann_feasible_claimed") is True for row in rows):
        reasons.append("hybrid_astar_ackermann_claimed")
    return reasons


def _boundary_rejections(config: dict[str, Any], platform_contract: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for source_name, source in (("config", config), ("platform_contract", platform_contract.get("boundary", {}))):
        for field in BOUNDARY_FIELDS:
            value = source.get(field)
            if value not in (False, 0, 0.0, None):
                reasons.append(f"{source_name}_{field}_enabled")
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("config_canary_traffic_fraction_nonzero")
    if float(platform_contract.get("boundary", {}).get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("platform_contract_canary_traffic_fraction_nonzero")
    return reasons


def _input_rejections(
    config: dict[str, Any],
    repo_root: Path,
    platform_contract: dict[str, Any],
    platform_hash: str,
) -> tuple[list[str], dict[str, Any] | None]:
    reasons: list[str] = []
    summary: dict[str, Any] | None = None
    if not bool(config.get("hybrid_astar_pose_path_cost_enabled")):
        reasons.append("hybrid_astar_pose_path_cost_not_enabled")
    if not config.get("candidate_rows"):
        reasons.append("candidate_rows_missing")
    if not isinstance(config.get("current_pose"), list) or len(config.get("current_pose")) < 3:
        reasons.append("current_pose_missing")
    if not str(config.get("current_pose_provenance") or "").strip():
        reasons.append("current_pose_provenance_missing")
    if not str(config.get("grid", {}).get("slope_obstacle_source_hash") or "").strip():
        reasons.append("slope_obstacle_source_hash_missing")
    stage24_0_root = _resolve_path(Path(str(config.get("stage24_0_root") or "")), repo_root)
    summary_path = stage24_0_root / "xunce-stage24-0-summary.json"
    if not summary_path.is_file():
        reasons.append("stage24_0_summary_missing")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "passed":
            reasons.append("stage24_0_not_passed")
        if summary.get("next_required_change") != "run_stage24_1_hybrid_astar_candidate_path_cost_integration":
            reasons.append("stage24_0_route_mismatch")
        if summary.get("default_astar_replaced") not in (False, 0, 0.0, None):
            reasons.append("stage24_0_default_astar_replaced")
        if not summary.get("platform_contract_hash"):
            reasons.append("stage24_0_platform_contract_hash_missing")
        elif summary.get("platform_contract_hash") != platform_hash:
            reasons.append("stage24_0_platform_contract_hash_mismatch")
        platform_slope = float(platform_contract["terrain_policy"]["max_traversable_slope_deg"])
        if summary.get("max_traversable_slope_deg") is None:
            reasons.append("stage24_0_max_traversable_slope_missing")
        elif float(summary["max_traversable_slope_deg"]) != platform_slope:
            reasons.append("stage24_0_max_traversable_slope_mismatch")
    return reasons, summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = _resolve_path(path, repo_root)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    return payload


def _load_platform_contract(path_value: str, repo_root: Path) -> tuple[dict[str, Any], str]:
    path = _resolve_path(Path(path_value), repo_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return payload, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _render_report(summary: dict[str, Any], audit: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage24.1 Hybrid A* Candidate Path Cost Integration",
            "",
            f"- status: {summary['status']}",
            f"- next_required_change: {summary['next_required_change']}",
            f"- path_cost_source_recommendation: {summary['path_cost_source_recommendation']}",
            f"- hybrid_astar_candidate_row_count: {summary['hybrid_astar_candidate_row_count']}",
            f"- hybrid_astar_reachable_count: {summary['hybrid_astar_reachable_count']}",
            f"- hybrid_vs_grid_cost_material: {summary['hybrid_vs_grid_cost_material']}",
            f"- material_cost_delta_count: {audit['material_cost_delta_count']}",
            f"- default_astar_replaced: {summary['default_astar_replaced']}",
            "",
            "Stage24.1 is audit-only path-cost integration evidence. It does not replace the default A* planner, publish checkpoints, or run PPO.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
