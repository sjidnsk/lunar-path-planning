from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los import (
    OBSTACLE_SOURCE_AUDIT_FILE as STAGE23_0A_OBSTACLE_SOURCE_AUDIT_FILE,
    SUMMARY_FILE as STAGE23_0A_SUMMARY_FILE,
    run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los,
)
from xunce_platform_contract import apply_stage23_platform_defaults


CONFIG_SCHEMA_VERSION = "xunce-stage23-1-slope-derived-obstacle-source-for-endpoint-theta-los-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage23-1-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage23-1-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage23-1-manifest/v1"
STAGE_ID = "xunce-stage23-1-slope-derived-obstacle-source-for-endpoint-theta-los"

DEFAULT_CONFIG = "configs/xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage23_endpoint_obstacle_aware_theta_sensor_coverage/"
    "outputs/path_feedback_batch_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los_v1"
)

SUMMARY_FILE = "xunce-stage23-1-summary.json"
SLOPE_AUDIT_FILE = "xunce-stage23-1-slope-source-audit.json"
RERUN_STAGE23_0A_SUMMARY_FILE = "xunce-stage23-1-rerun-stage23-0a-summary.json"
RERUN_STAGE23_0_SUMMARY_FILE = "xunce-stage23-1-rerun-stage23-0-summary.json"
ROUTING_FILE = "xunce-stage23-1-next-stage-routing.json"
REPORT_FILE = "xunce-stage23-1-report.md"
MANIFEST_FILE = "xunce-stage23-1-manifest.json"
STAGE23_0A_CONFIG_FILE = "xunce-stage23-1-stage23-0a-config.json"
STAGE23_0_LOS_AUDIT_FILE = "xunce-stage23-0-obstacle-los-audit.jsonl"

BOUNDARY_FIELDS = (
    "stage23_1_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage23.1 slope-derived obstacle source for endpoint theta LOS.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los(
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

    stage23_0a_root = _prepare_stage23_0a_root(config, output_root=output_root, repo_root=repo_root)
    stage23_0a_summary = _read_json(stage23_0a_root / STAGE23_0A_SUMMARY_FILE)
    slope_audit = _slope_source_audit(stage23_0a_root, stage23_0a_summary)
    stage23_0_summary = _read_json(stage23_0a_root / "s23_0" / "xunce-stage23-0-summary.json")
    status, route, route_reason, blockers = _route(boundary_reasons, stage23_0a_summary, slope_audit, stage23_0_summary)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blockers,
        "stage23_0a_root": str(stage23_0a_root),
        "stage23_0a_status": stage23_0a_summary.get("status"),
        "stage23_0a_next_required_change": stage23_0a_summary.get("next_required_change"),
        "stage23_0_status": stage23_0_summary.get("status"),
        "stage23_0_next_required_change": stage23_0_summary.get("next_required_change"),
        "platform_contract_id": config.get("platform_contract_id"),
        "platform_contract_hash": config.get("platform_contract_hash"),
        "platform_max_climb_deg": float(config.get("platform_max_climb_deg", 30.0)),
        "max_traversable_slope_deg": float(config.get("max_traversable_slope_deg", 30.0)),
        "slope_sensitivity_thresholds_deg": config.get("slope_sensitivity_thresholds_deg", [20.0, 30.0]),
        "obstacle_source_count": int(stage23_0a_summary.get("obstacle_source_count", 0) or 0),
        "obstacle_source_kind_counts": slope_audit["obstacle_source_kind_counts"],
        "slope_source_count": slope_audit["slope_source_count"],
        "slope_blocked_cell_count_total": slope_audit["slope_blocked_cell_count_total"],
        "candidate_obstacle_source_missing_count": int(stage23_0a_summary.get("candidate_obstacle_source_missing_count", 0) or 0),
        "candidate_obstacle_source_hash_mismatch_count": int(
            stage23_0a_summary.get("candidate_obstacle_source_hash_mismatch_count", 0) or 0
        ),
        "stage23_0_los_audit_row_count": int(stage23_0a_summary.get("stage23_0_los_audit_row_count", 0) or 0),
        "stage23_0_obstacle_occlusion_material_to_coverage": bool(
            stage23_0a_summary.get("stage23_0_obstacle_occlusion_material_to_coverage", False)
        ),
        "slope_los_audit_row_count": slope_audit["slope_los_audit_row_count"],
        "slope_material_occlusion_viewpoint_count": slope_audit["slope_material_occlusion_viewpoint_count"],
        "slope_max_visibility_removed_by_obstacle_count": slope_audit[
            "slope_max_visibility_removed_by_obstacle_count"
        ],
        "stage23_1_authorized": False,
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
        "stage23_1_authorized": False,
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
        "slope_source_audit": str(output_root / SLOPE_AUDIT_FILE),
        "rerun_stage23_0a_summary": str(output_root / RERUN_STAGE23_0A_SUMMARY_FILE),
        "rerun_stage23_0_summary": str(output_root / RERUN_STAGE23_0_SUMMARY_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / SLOPE_AUDIT_FILE, slope_audit)
    _write_json(output_root / RERUN_STAGE23_0A_SUMMARY_FILE, stage23_0a_summary)
    _write_json(output_root / RERUN_STAGE23_0_SUMMARY_FILE, stage23_0_summary)
    _write_json(output_root / ROUTING_FILE, routing)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    _write_json(output_root / MANIFEST_FILE, manifest)
    return summary


def _prepare_stage23_0a_root(config: dict[str, Any], *, output_root: Path, repo_root: Path) -> Path:
    if not bool(config.get("run_stage23_0a", True)):
        root = config.get("stage23_0a_root")
        if not root:
            raise ValueError("stage23_0a_root is required when run_stage23_0a=false")
        return _resolve_path(Path(str(root)), repo_root)
    base_config = _read_json(_resolve_path(Path(str(config["stage23_0a_config"])), repo_root))
    base_config.update(
        {
            "derive_slope_blocked_cells_from_sidecar_dem": True,
            "platform_contract": config.get("platform_contract"),
            "platform_contract_id": config.get("platform_contract_id"),
            "platform_contract_hash": config.get("platform_contract_hash"),
            "platform_max_climb_deg": float(config.get("platform_max_climb_deg", 30.0)),
            "max_traversable_slope_deg": float(config.get("max_traversable_slope_deg", 30.0)),
            "slope_sensitivity_thresholds_deg": config.get("slope_sensitivity_thresholds_deg", [20.0, 30.0]),
            "stage23_0a_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    for key in (
        "required_scenario_count",
        "rollout_steps",
        "dynamic_max_candidates_per_step",
        "dynamic_proposal_pool_limit_per_step",
        "sensor_range_cells",
        "sensor_fov_deg",
        "theta_bin_count",
        "theta_step_deg",
    ):
        if key in config:
            base_config[key] = config[key]
    stage23_0a_config = output_root / STAGE23_0A_CONFIG_FILE
    _write_json(stage23_0a_config, base_config)
    # Keep the nested work directory very short. The public Stage23.1 output root
    # is already long, and high-fidelity artifact names can otherwise exceed the
    # Windows legacy 260-character path boundary.
    stage23_0a_root = output_root / str(config.get("stage23_0a_output_subdir", "a"))
    run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los(
        config_path=stage23_0a_config,
        output_root=stage23_0a_root,
        repo_root=repo_root,
    )
    return stage23_0a_root


def _slope_source_audit(stage23_0a_root: Path, stage23_0a_summary: dict[str, Any]) -> dict[str, Any]:
    path = stage23_0a_root / STAGE23_0A_OBSTACLE_SOURCE_AUDIT_FILE
    payload = _read_json(path)
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    kind_counts = Counter(str(row.get("obstacle_source_kind") or "unknown") for row in sources if isinstance(row, dict))
    slope_sources = [
        row for row in sources if isinstance(row, dict) and row.get("obstacle_source_kind") == "slope_blocked_as_obstacle_proxy"
    ]
    los_audit = _slope_los_material_audit(stage23_0a_root)
    return {
        "schema_version": "xunce-stage23-1-slope-source-audit/v1",
        "stage23_0a_obstacle_source_audit": str(path) if path.exists() else None,
        "source_count": int(stage23_0a_summary.get("obstacle_source_count", len(sources)) or 0),
        "obstacle_source_kind_counts": dict(kind_counts or stage23_0a_summary.get("obstacle_source_kind_counts", {}) or {}),
        "slope_source_count": len(slope_sources),
        "slope_blocked_cell_count_total": sum(int(row.get("obstacle_cell_count", 0) or 0) for row in slope_sources),
        **los_audit,
        "slope_source_examples": slope_sources[:5],
    }


def _slope_los_material_audit(stage23_0a_root: Path) -> dict[str, Any]:
    path = stage23_0a_root / "s23_0" / STAGE23_0_LOS_AUDIT_FILE
    rows = _read_jsonl(path)
    slope_rows = [row for row in rows if row.get("obstacle_source_kind") == "slope_blocked_as_obstacle_proxy"]
    material_rows = [row for row in slope_rows if bool(row.get("coverage_changed_by_obstacle", False))]
    removed_counts = [int(row.get("visibility_removed_by_obstacle_count", 0) or 0) for row in slope_rows]
    return {
        "stage23_0_los_audit": str(path) if path.exists() else None,
        "slope_los_audit_row_count": len(slope_rows),
        "slope_material_occlusion_viewpoint_count": len(material_rows),
        "slope_max_visibility_removed_by_obstacle_count": max(removed_counts) if removed_counts else 0,
        "slope_mean_visibility_removed_by_obstacle_count": (sum(removed_counts) / len(removed_counts))
        if removed_counts
        else 0.0,
    }


def _route(
    boundary_reasons: list[str],
    stage23_0a_summary: dict[str, Any],
    slope_audit: dict[str, Any],
    stage23_0_summary: dict[str, Any],
) -> tuple[str, str, str, list[str]]:
    if boundary_reasons:
        return "failed", "resolve_stage23_1_boundary_rejections", "boundary_rejection", sorted(boundary_reasons)
    if int(slope_audit.get("source_count", 0)) <= 0 or int(slope_audit.get("slope_source_count", 0)) <= 0:
        return (
            "failed",
            "rerun_stage23_1_required_dem_sources",
            "slope_obstacle_source_absent",
            ["slope_obstacle_source_absent"],
        )
    missing = int(stage23_0a_summary.get("candidate_obstacle_source_missing_count", 0) or 0)
    mismatch = int(stage23_0a_summary.get("candidate_obstacle_source_hash_mismatch_count", 0) or 0)
    los_rows = int(stage23_0a_summary.get("stage23_0_los_audit_row_count", 0) or 0)
    slope_los_rows = int(slope_audit.get("slope_los_audit_row_count", 0) or 0)
    if missing > 0 or mismatch > 0 or los_rows <= 0 or slope_los_rows <= 0:
        blockers = []
        if missing > 0:
            blockers.append("candidate_obstacle_source_missing")
        if mismatch > 0:
            blockers.append("candidate_obstacle_source_hash_mismatch")
        if los_rows <= 0:
            blockers.append("stage23_0_los_audit_missing")
        if slope_los_rows <= 0:
            blockers.append("slope_stage23_0_los_audit_missing")
        return "failed", "repair_stage23_1_slope_source_lineage", "slope_source_lineage_invalid", sorted(blockers)
    if stage23_0a_summary.get("status") != "passed" or stage23_0_summary.get("status") != "passed":
        return (
            "failed",
            "repair_stage23_1_slope_obstacle_source_contract",
            "stage23_0a_or_stage23_0_not_passed",
            ["stage23_0a_or_stage23_0_not_passed"],
        )
    if int(slope_audit.get("slope_material_occlusion_viewpoint_count", 0) or 0) > 0:
        return (
            "passed",
            "implement_stage23_2_slope_obstacle_aware_theta_reward_contract",
            "slope_obstacle_occlusion_material_to_coverage",
            [],
        )
    return (
        "passed",
        "document_slope_obstacle_occlusion_audit_only",
        "slope_obstacle_occlusion_not_material_to_coverage",
        [],
    )


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in BOUNDARY_FIELDS:
        if bool(config.get(field, False)):
            reasons.append(f"{field}_must_be_false")
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction_must_be_zero")
    return reasons


def _load_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    path = _resolve_path(config_path, repo_root)
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    payload.setdefault("stage23_0a_config", "configs/xunce_stage23_0a_materialize_obstacle_sources_for_theta_los_v1.json")
    payload.setdefault("stage23_0a_output_subdir", "a")
    payload.setdefault("run_stage23_0a", True)
    return apply_stage23_platform_defaults(payload, repo_root=repo_root)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage23.1 Slope-Derived Obstacle Source For Endpoint Theta LOS",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- platform_contract_id: `{summary.get('platform_contract_id')}`",
            f"- platform_max_climb_deg: `{summary.get('platform_max_climb_deg')}`",
            f"- max_traversable_slope_deg: `{summary['max_traversable_slope_deg']}`",
            f"- slope_sensitivity_thresholds_deg: `{summary.get('slope_sensitivity_thresholds_deg')}`",
            f"- slope_source_count: `{summary['slope_source_count']}`",
            f"- slope_blocked_cell_count_total: `{summary['slope_blocked_cell_count_total']}`",
            f"- stage23_0_los_audit_row_count: `{summary['stage23_0_los_audit_row_count']}`",
            f"- slope_los_audit_row_count: `{summary['slope_los_audit_row_count']}`",
            "- slope_material_occlusion_viewpoint_count: "
            f"`{summary['slope_material_occlusion_viewpoint_count']}`",
            "- stage23_0_obstacle_occlusion_material_to_coverage: "
            f"`{summary['stage23_0_obstacle_occlusion_material_to_coverage']}`",
            "",
            "`slope_blocked_cells` 是 slope-derived obstacle proxy，不是真实岩石、墙体或裂缝标注。",
            "本阶段不训练、不发布、不替换 default policy、不连接 executor、不启动 canary。",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
