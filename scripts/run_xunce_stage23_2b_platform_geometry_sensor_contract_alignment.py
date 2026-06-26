from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from run_xunce_stage23_2a_high_resolution_terrain_data_prepare import (
        STAGE23_1_SMOKE_SUMMARY_FILE,
        SUMMARY_FILE as STAGE23_2A_SUMMARY_FILE,
        run_xunce_stage23_2a_high_resolution_terrain_data_prepare,
    )
    from xunce_platform_contract import apply_stage23_platform_defaults, load_platform_contract, platform_lineage
except ModuleNotFoundError:  # pragma: no cover
    from scripts.run_xunce_stage23_2a_high_resolution_terrain_data_prepare import (
        STAGE23_1_SMOKE_SUMMARY_FILE,
        SUMMARY_FILE as STAGE23_2A_SUMMARY_FILE,
        run_xunce_stage23_2a_high_resolution_terrain_data_prepare,
    )
    from scripts.xunce_platform_contract import apply_stage23_platform_defaults, load_platform_contract, platform_lineage


CONFIG_SCHEMA_VERSION = "xunce-stage23-2b-platform-geometry-sensor-contract-alignment-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage23-2b-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage23-2b-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage23-2b-manifest/v1"
STAGE_ID = "xunce-stage23-2b-platform-geometry-sensor-contract-alignment"

DEFAULT_CONFIG = "configs/xunce_stage23_2b_platform_geometry_sensor_contract_alignment_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage23_platform_alignment/"
    "outputs/path_feedback_batch_xunce_stage23_2b_platform_geometry_sensor_contract_alignment_v1"
)

SUMMARY_FILE = "xunce-stage23-2b-summary.json"
PLATFORM_AUDIT_FILE = "xunce-stage23-2b-platform-contract-audit.json"
THRESHOLD_COMPARISON_FILE = "xunce-stage23-2b-slope-threshold-comparison.json"
RERUN_STAGE23_2A_SUMMARY_FILE = "xunce-stage23-2b-rerun-stage23-2a-summary.json"
RERUN_STAGE23_1_SUMMARY_FILE = "xunce-stage23-2b-rerun-stage23-1-summary.json"
ROUTING_FILE = "xunce-stage23-2b-next-stage-routing.json"
REPORT_FILE = "xunce-stage23-2b-report.md"
MANIFEST_FILE = "xunce-stage23-2b-manifest.json"

BOUNDARY_FIELDS = (
    "stage23_2b_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage23.2B platform geometry/sensor contract alignment.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage23_2b_platform_geometry_sensor_contract_alignment(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage23_2b_platform_geometry_sensor_contract_alignment(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path, repo_root)
    contract = load_platform_contract(config, repo_root=repo_root)
    lineage = platform_lineage(contract)
    boundary_reasons = _boundary_rejections(config)
    contract_audit = _platform_contract_audit(contract, lineage, config)

    default_threshold = float(config.get("default_threshold_deg", lineage["max_traversable_slope_deg"]))
    sensitivity_threshold = float(config.get("sensitivity_threshold_deg", 20.0))

    default_root = output_root / str(config.get("stage23_2a_default_output_subdir", "s23_2a_platform_30"))
    sensitivity_root = output_root / "s23_2a_sensitivity_20"
    sensitivity_root = output_root / str(config.get("stage23_2a_sensitivity_output_subdir", "s23_2a_sensitivity_20"))
    default_2a, default_1 = _stage23_2a_pair(
        config,
        output_root=default_root,
        repo_root=repo_root,
        threshold_deg=default_threshold,
        execute=bool(config.get("run_stage23_2a_smoke", True)) and not boundary_reasons,
        fallback_root=config.get("stage23_2a_default_root"),
    )
    sensitivity_2a, sensitivity_1 = _stage23_2a_pair(
        config,
        output_root=sensitivity_root,
        repo_root=repo_root,
        threshold_deg=sensitivity_threshold,
        execute=bool(config.get("run_sensitivity_smoke", config.get("run_stage23_2a_smoke", True))) and not boundary_reasons,
        fallback_root=config.get("stage23_2a_sensitivity_root"),
    )
    comparison = _threshold_comparison(
        default_threshold=default_threshold,
        sensitivity_threshold=sensitivity_threshold,
        default_2a=default_2a,
        default_1=default_1,
        sensitivity_2a=sensitivity_2a,
        sensitivity_1=sensitivity_1,
    )
    status, route, route_reason, blockers = _route(
        boundary_reasons=boundary_reasons,
        contract_audit=contract_audit,
        default_2a=default_2a,
        default_1=default_1,
        comparison=comparison,
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blockers,
        "platform_contract_id": lineage["platform_contract_id"],
        "platform_contract_hash": lineage["platform_contract_hash"],
        "platform_max_climb_deg": lineage["platform_max_climb_deg"],
        "max_traversable_slope_deg": default_threshold,
        "sensitivity_threshold_deg": sensitivity_threshold,
        "default_stage23_2a_status": default_2a.get("status"),
        "default_stage23_1_status": default_1.get("status"),
        "default_slope_blocked_cell_count_total": default_2a.get("slope_blocked_cell_count_total", 0),
        "default_slope_material_occlusion_viewpoint_count": default_1.get("slope_material_occlusion_viewpoint_count", 0),
        "sensitivity_slope_blocked_cell_count_total": sensitivity_2a.get("slope_blocked_cell_count_total", 0),
        "sensitivity_slope_material_occlusion_viewpoint_count": sensitivity_1.get("slope_material_occlusion_viewpoint_count", 0),
        "stage23_2b_authorized": False,
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
        "stage23_2b_authorized": False,
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
        "platform_contract_audit": str(output_root / PLATFORM_AUDIT_FILE),
        "slope_threshold_comparison": str(output_root / THRESHOLD_COMPARISON_FILE),
        "rerun_stage23_2a_summary": str(output_root / RERUN_STAGE23_2A_SUMMARY_FILE),
        "rerun_stage23_1_summary": str(output_root / RERUN_STAGE23_1_SUMMARY_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / PLATFORM_AUDIT_FILE, contract_audit)
    _write_json(output_root / THRESHOLD_COMPARISON_FILE, comparison)
    _write_json(output_root / RERUN_STAGE23_2A_SUMMARY_FILE, default_2a)
    _write_json(output_root / RERUN_STAGE23_1_SUMMARY_FILE, default_1)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _stage23_2a_pair(
    config: dict[str, Any],
    *,
    output_root: Path,
    repo_root: Path,
    threshold_deg: float,
    execute: bool,
    fallback_root: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if execute:
        base = _read_json(_resolve_path(Path(str(config["stage23_2a_config"])), repo_root))
        base.update(
            {
                "platform_contract": config.get("platform_contract"),
                "platform_contract_id": config.get("platform_contract_id"),
                "platform_contract_hash": config.get("platform_contract_hash"),
                "platform_max_climb_deg": config.get("platform_max_climb_deg"),
                "max_traversable_slope_deg": float(threshold_deg),
                "slope_sensitivity_thresholds_deg": config.get("slope_sensitivity_thresholds_deg", [20.0, 30.0]),
                "run_stage23_1_smoke": bool(config.get("run_stage23_1_smoke", True)),
                "stage23_2a_authorized": False,
                "runs_new_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "canary_traffic_fraction": 0.0,
            }
        )
        output_root.mkdir(parents=True, exist_ok=True)
        config_path = output_root / "xunce-stage23-2b-stage23-2a-config.json"
        _write_json(config_path, base)
        summary = run_xunce_stage23_2a_high_resolution_terrain_data_prepare(
            config_path=config_path,
            output_root=output_root,
            repo_root=repo_root,
        )
    else:
        if not fallback_root:
            return {}, {}
        output_root = _resolve_path(Path(str(fallback_root)), repo_root)
        summary = _read_json(output_root / STAGE23_2A_SUMMARY_FILE)
    stage23_1_summary = _read_json(output_root / STAGE23_1_SMOKE_SUMMARY_FILE)
    return summary, stage23_1_summary


def _platform_contract_audit(contract: dict[str, Any], lineage: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if lineage["platform_contract_id"] != "agilex_scout_mini_piper":
        issues.append("unexpected_platform_contract_id")
    if float(lineage["platform_max_climb_deg"]) != 30.0:
        issues.append("platform_max_climb_not_30_deg")
    if float(lineage["max_traversable_slope_deg"]) != 30.0:
        issues.append("max_traversable_slope_not_30_deg")
    if float(config.get("default_threshold_deg", 30.0)) != 30.0:
        issues.append("default_threshold_not_30_deg")
    return {
        "schema_version": "xunce-stage23-2b-platform-contract-audit/v1",
        "status": "passed" if not issues else "failed",
        "issues": issues,
        **lineage,
        "body_length_m": contract.get("base", {}).get("body_length_m"),
        "body_width_m": contract.get("base", {}).get("body_width_m"),
        "ground_clearance_m": contract.get("base", {}).get("ground_clearance_m"),
        "livox_mid360": contract.get("sensors", {}).get("livox_mid360", {}),
        "orbbec_dabai_camera": contract.get("sensors", {}).get("orbbec_dabai_camera", {}),
    }


def _threshold_comparison(
    *,
    default_threshold: float,
    sensitivity_threshold: float,
    default_2a: dict[str, Any],
    default_1: dict[str, Any],
    sensitivity_2a: dict[str, Any],
    sensitivity_1: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage23-2b-slope-threshold-comparison/v1",
        "default_threshold_deg": default_threshold,
        "sensitivity_threshold_deg": sensitivity_threshold,
        "default_slope_blocked_cell_count_total": int(default_2a.get("slope_blocked_cell_count_total", 0) or 0),
        "sensitivity_slope_blocked_cell_count_total": int(sensitivity_2a.get("slope_blocked_cell_count_total", 0) or 0),
        "default_slope_los_audit_row_count": int(default_1.get("slope_los_audit_row_count", 0) or 0),
        "sensitivity_slope_los_audit_row_count": int(sensitivity_1.get("slope_los_audit_row_count", 0) or 0),
        "default_slope_material_occlusion_viewpoint_count": int(
            default_1.get("slope_material_occlusion_viewpoint_count", 0) or 0
        ),
        "sensitivity_slope_material_occlusion_viewpoint_count": int(
            sensitivity_1.get("slope_material_occlusion_viewpoint_count", 0) or 0
        ),
    }


def _route(
    *,
    boundary_reasons: list[str],
    contract_audit: dict[str, Any],
    default_2a: dict[str, Any],
    default_1: dict[str, Any],
    comparison: dict[str, Any],
) -> tuple[str, str, str, list[str]]:
    if boundary_reasons:
        return "failed", "resolve_stage23_2b_boundary_rejections", "boundary_rejection", sorted(boundary_reasons)
    if contract_audit.get("status") != "passed":
        return "failed", "repair_stage23_2b_platform_contract", "platform_contract_invalid", list(contract_audit.get("issues", []))
    if default_2a.get("status") != "passed" or default_1.get("status") != "passed":
        return "failed", "repair_stage23_2b_platform_aligned_smoke", "platform_aligned_smoke_failed", [
            str(default_2a.get("next_required_change") or default_1.get("next_required_change") or "stage23_2b_smoke_failed")
        ]
    if int(comparison.get("default_slope_material_occlusion_viewpoint_count", 0) or 0) > 0:
        return "passed", "implement_stage23_2_slope_obstacle_aware_theta_reward_contract", "platform_aligned_slope_los_material", []
    return "passed", "document_platform_aligned_slope_occlusion_audit_only", "platform_aligned_slope_los_not_material", []


def _load_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(config_path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    payload.setdefault("stage23_2a_config", "configs/xunce_stage23_2a_high_resolution_terrain_data_ingestion_v1.json")
    payload.setdefault("stage23_1_config", "configs/xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los_v1.json")
    payload.setdefault("run_stage23_2a_smoke", True)
    payload.setdefault("run_sensitivity_smoke", payload.get("run_stage23_2a_smoke", True))
    payload.setdefault("run_stage23_1_smoke", True)
    payload.setdefault("default_threshold_deg", 30.0)
    payload.setdefault("sensitivity_threshold_deg", 20.0)
    payload = apply_stage23_platform_defaults(payload, repo_root=repo_root)
    if float(payload["max_traversable_slope_deg"]) != 30.0:
        payload["max_traversable_slope_deg"] = 30.0
    return payload


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in BOUNDARY_FIELDS:
        if bool(config.get(field, False)):
            reasons.append(f"{field}_must_be_false")
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction_must_be_zero")
    return reasons


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage23.2B Platform Geometry Sensor Contract Alignment",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- platform_contract_id: `{summary['platform_contract_id']}`",
            f"- platform_max_climb_deg: `{summary['platform_max_climb_deg']}`",
            f"- max_traversable_slope_deg: `{summary['max_traversable_slope_deg']}`",
            f"- sensitivity_threshold_deg: `{summary['sensitivity_threshold_deg']}`",
            f"- default_slope_blocked_cell_count_total: `{summary['default_slope_blocked_cell_count_total']}`",
            "- default_slope_material_occlusion_viewpoint_count: "
            f"`{summary['default_slope_material_occlusion_viewpoint_count']}`",
            "",
            "Stage23 hard slope obstacle 默认阈值严格对齐 Scout Mini 平台最大爬坡能力 30°。",
            "20° 只作为 sensitivity/audit 基线，不再作为默认 hard gate。",
            "",
        ]
    )


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
