from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:  # pragma: no cover - script execution path
    from run_xunce_high_fidelity_exploration_coverage_comparison import (
        run_xunce_high_fidelity_exploration_coverage_comparison,
    )
    from run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract import (
        run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract,
    )
    from xunce_platform_contract import apply_stage23_platform_defaults
except ModuleNotFoundError:  # pragma: no cover
    from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
        run_xunce_high_fidelity_exploration_coverage_comparison,
    )
    from scripts.run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract import (
        run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract,
    )
    from scripts.xunce_platform_contract import apply_stage23_platform_defaults


CONFIG_SCHEMA_VERSION = "xunce-stage23-0a-materialize-obstacle-sources-for-theta-los-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage23-0a-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage23-0a-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage23-0a-manifest/v1"
STAGE_ID = "xunce-stage23-0a-materialize-obstacle-sources-for-theta-los"

DEFAULT_CONFIG = "configs/xunce_stage23_0a_materialize_obstacle_sources_for_theta_los_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage23_endpoint_obstacle_aware_theta_sensor_coverage/"
    "outputs/path_feedback_batch_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los_v1"
)
DEFAULT_HIGH_FIDELITY_CONFIG = "configs/xunce_high_fidelity_exploration_coverage_comparison_stage18_9_strict_v3.json"

SUMMARY_FILE = "xunce-stage23-0a-summary.json"
OBSTACLE_SOURCE_AUDIT_FILE = "xunce-stage23-0a-obstacle-source-audit.json"
CANDIDATE_LINKAGE_FILE = "xunce-stage23-0a-candidate-audit-source-linkage.json"
RERUN_STAGE23_0_SUMMARY_FILE = "xunce-stage23-0a-rerun-stage23-0-summary.json"
ROUTING_FILE = "xunce-stage23-0a-next-stage-routing.json"
REPORT_FILE = "xunce-stage23-0a-report.md"
MANIFEST_FILE = "xunce-stage23-0a-manifest.json"

HF_OBSTACLE_SOURCES_FILE = "xunce-exploration-coverage-obstacle-sources.json"
HF_CANDIDATE_AUDIT_FILE = "xunce-exploration-coverage-candidate-metric-audit.jsonl"

ROUTE_MISSING_SOURCE = "repair_stage23_map_obstacle_source_export"
ROUTE_PROXY_REVIEW = "review_blocked_as_obstacle_proxy_semantics"
ROUTE_LINKAGE = "repair_stage23_0a_candidate_obstacle_source_linkage"
ROUTE_STAGE23_1 = "implement_stage23_1_obstacle_aware_theta_reward_contract"
ROUTE_AUDIT_ONLY = "document_obstacle_occlusion_audit_only"
ROUTE_BOUNDARY = "resolve_stage23_0a_boundary_rejections"

BOUNDARY_FIELDS = (
    "stage23_0a_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage23.0A obstacle source materialization for theta LOS.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los(
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

    high_fidelity_root = _prepare_high_fidelity_root(config, output_root=output_root, repo_root=repo_root)
    obstacle_audit = _obstacle_source_audit(high_fidelity_root)
    linkage_audit = _candidate_linkage_audit(high_fidelity_root, obstacle_audit)
    rerun_summary = _rerun_stage23_0(config, high_fidelity_root=high_fidelity_root, output_root=output_root, repo_root=repo_root)
    status, route, route_reason = _route(boundary_reasons, obstacle_audit, linkage_audit, rerun_summary)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": sorted(set(boundary_reasons + _blocking_reasons(obstacle_audit, linkage_audit, rerun_summary))),
        "high_fidelity_root": str(high_fidelity_root),
        "platform_contract_id": config.get("platform_contract_id"),
        "platform_contract_hash": config.get("platform_contract_hash"),
        "platform_max_climb_deg": config.get("platform_max_climb_deg"),
        "max_traversable_slope_deg": config.get("max_traversable_slope_deg"),
        "obstacle_source_count": obstacle_audit["source_count"],
        "obstacle_source_kind_counts": obstacle_audit["obstacle_source_kind_counts"],
        "only_blocked_proxy_sources_available": obstacle_audit["only_blocked_proxy_sources_available"],
        "only_proxy_sources_available": obstacle_audit["only_proxy_sources_available"],
        "candidate_row_count": linkage_audit["candidate_row_count"],
        "candidate_obstacle_source_missing_count": linkage_audit["candidate_obstacle_source_missing_count"],
        "candidate_obstacle_source_hash_mismatch_count": linkage_audit["candidate_obstacle_source_hash_mismatch_count"],
        "stage23_0_status": rerun_summary.get("status"),
        "stage23_0_next_required_change": rerun_summary.get("next_required_change"),
        "stage23_0_los_audit_row_count": rerun_summary.get("los_audit_row_count", 0),
        "stage23_0_obstacle_occlusion_material_to_coverage": rerun_summary.get("obstacle_occlusion_material_to_coverage", False),
        "stage23_0a_authorized": False,
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
        "stage23_0a_authorized": False,
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
        "obstacle_source_audit": str(output_root / OBSTACLE_SOURCE_AUDIT_FILE),
        "candidate_audit_source_linkage": str(output_root / CANDIDATE_LINKAGE_FILE),
        "rerun_stage23_0_summary": str(output_root / RERUN_STAGE23_0_SUMMARY_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / OBSTACLE_SOURCE_AUDIT_FILE, obstacle_audit)
    _write_json(output_root / CANDIDATE_LINKAGE_FILE, linkage_audit)
    _write_json(output_root / RERUN_STAGE23_0_SUMMARY_FILE, rerun_summary)
    _write_json(output_root / ROUTING_FILE, routing)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    _write_json(output_root / MANIFEST_FILE, manifest)
    return summary


def _prepare_high_fidelity_root(config: dict[str, Any], *, output_root: Path, repo_root: Path) -> Path:
    existing_root = config.get("high_fidelity_root")
    if existing_root and not bool(config.get("run_high_fidelity_smoke", False)):
        return _resolve_path(Path(str(existing_root)), repo_root)
    high_fidelity_output_root = output_root / str(config.get("high_fidelity_output_subdir", "hf"))
    high_fidelity_output_root.mkdir(parents=True, exist_ok=True)
    hf_config = _resolve_path(Path(str(config.get("high_fidelity_config", DEFAULT_HIGH_FIDELITY_CONFIG))), repo_root)
    overrides = {
        "required_scenario_count": int(config.get("required_scenario_count", 2)),
        "rollout_steps": int(config.get("rollout_steps", 4)),
        "theta_aware_candidate_viewpoints_enabled": True,
        "theta_bin_count": int(config.get("theta_bin_count", 8)),
        "theta_step_deg": int(config.get("theta_step_deg", 45)),
        "sensor_fov_deg": float(config.get("sensor_fov_deg", 90.0)),
        "sensor_range_cells": int(config.get("sensor_range_cells", config.get("coverage_radius_cells", 3))),
        "emit_candidate_metric_audit": True,
        "emit_obstacle_source_audit": True,
        "obstacle_occlusion_enabled": bool(config.get("obstacle_occlusion_enabled", False)),
        "no_go_blocks_los": bool(config.get("no_go_blocks_los", False)),
        "derive_slope_blocked_cells_from_sidecar_dem": bool(
            config.get("derive_slope_blocked_cells_from_sidecar_dem", False)
        ),
        "platform_contract": config.get("platform_contract"),
        "platform_contract_id": config.get("platform_contract_id"),
        "platform_contract_hash": config.get("platform_contract_hash"),
        "platform_max_climb_deg": float(config.get("platform_max_climb_deg", 30.0)),
        "max_traversable_slope_deg": float(config.get("max_traversable_slope_deg", 30.0)),
        "slope_sensitivity_thresholds_deg": config.get("slope_sensitivity_thresholds_deg", [20.0, 30.0]),
    }
    for key in (
        "source_roi_expansion_root",
        "dynamic_max_candidates_per_step",
        "dynamic_proposal_pool_limit_per_step",
        "candidate_refresh_mode",
        "dynamic_candidate_validation_mode",
        "dynamic_candidate_generation_mode",
        "dynamic_candidate_selection_mode",
        "coverage_metric_mode",
    ):
        if key in config:
            overrides[key] = config[key]
    run_xunce_high_fidelity_exploration_coverage_comparison(
        config_path=hf_config,
        output_root=high_fidelity_output_root,
        repo_root=repo_root,
        config_overrides=overrides,
    )
    return high_fidelity_output_root


def _obstacle_source_audit(high_fidelity_root: Path) -> dict[str, Any]:
    path = high_fidelity_root / HF_OBSTACLE_SOURCES_FILE
    if not path.exists():
        return {
            "obstacle_source_artifact": None,
            "source_count": 0,
            "obstacle_source_kind_counts": {},
        "only_blocked_proxy_sources_available": False,
        "only_proxy_sources_available": False,
            "sources": [],
            "source_hashes": {},
        }
    payload = _read_json(path)
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    kind_counts = Counter(str(row.get("obstacle_source_kind") or "unknown") for row in sources if isinstance(row, dict))
    source_hashes = {
        str(row.get("obstacle_source_id")): str(row.get("obstacle_source_hash"))
        for row in sources
        if isinstance(row, dict) and row.get("obstacle_source_id")
    }
    return {
        "obstacle_source_artifact": str(path),
        "source_count": len(sources),
        "obstacle_source_kind_counts": dict(kind_counts),
        "only_blocked_proxy_sources_available": bool(sources and set(kind_counts) == {"blocked_as_obstacle_proxy"}),
        "only_proxy_sources_available": _only_proxy_sources_available(kind_counts),
        "sources": sources,
        "source_hashes": source_hashes,
    }


def _only_proxy_sources_available(kind_counts: Counter[str] | dict[str, int]) -> bool:
    if not kind_counts:
        return False
    proxy_kinds = {"blocked_as_obstacle_proxy", "slope_blocked_as_obstacle_proxy", "no_go_as_obstacle_proxy"}
    return set(kind_counts) <= proxy_kinds


def _candidate_linkage_audit(high_fidelity_root: Path, obstacle_audit: dict[str, Any]) -> dict[str, Any]:
    path = high_fidelity_root / HF_CANDIDATE_AUDIT_FILE
    if not path.exists():
        return {
            "candidate_audit": None,
            "candidate_row_count": 0,
            "candidate_obstacle_source_missing_count": 0,
            "candidate_obstacle_source_hash_mismatch_count": 0,
            "candidate_obstacle_source_available_count": 0,
        }
    source_hashes = obstacle_audit.get("source_hashes", {})
    missing = 0
    mismatch = 0
    available = 0
    rows = _read_jsonl(path)
    for row in rows:
        source_id = row.get("obstacle_source_id")
        source_hash = row.get("obstacle_source_hash")
        if row.get("obstacle_source_missing") is True or not source_id or not source_hash:
            missing += 1
            continue
        available += 1
        expected_hash = source_hashes.get(str(source_id))
        if not expected_hash or str(source_hash) != str(expected_hash):
            mismatch += 1
    return {
        "candidate_audit": str(path),
        "candidate_row_count": len(rows),
        "candidate_obstacle_source_missing_count": missing,
        "candidate_obstacle_source_hash_mismatch_count": mismatch,
        "candidate_obstacle_source_available_count": available,
    }


def _rerun_stage23_0(config: dict[str, Any], *, high_fidelity_root: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    stage23_config = {
        "schema_version": "xunce-stage23-0-endpoint-obstacle-aware-theta-sensor-coverage-contract-config/v1",
        "high_fidelity_root": str(high_fidelity_root),
        "stage22_5_root": str(config.get("stage22_5_root", "")),
        "max_audit_rows": int(config.get("max_audit_rows", 5000)),
        "stage22_reward_obstacle_occlusion_enabled": bool(config.get("stage22_reward_obstacle_occlusion_enabled", False)),
        "no_go_blocks_los": bool(config.get("no_go_blocks_los", False)),
        "stage23_0_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    path = output_root / "xunce-stage23-0a-rerun-stage23-0-config.json"
    _write_json(path, stage23_config)
    return run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract(
        config_path=path,
        output_root=output_root / "s23_0",
        repo_root=repo_root,
    )


def _route(
    boundary_reasons: list[str],
    obstacle_audit: dict[str, Any],
    linkage_audit: dict[str, Any],
    rerun_summary: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage23_0a_boundary_rejected"
    if int(obstacle_audit.get("source_count", 0)) <= 0:
        return "failed", ROUTE_MISSING_SOURCE, "stage23_0a_no_obstacle_sources"
    if int(linkage_audit.get("candidate_obstacle_source_missing_count", 0)) > 0 or int(linkage_audit.get("candidate_obstacle_source_hash_mismatch_count", 0)) > 0:
        return "failed", ROUTE_LINKAGE, "stage23_0a_candidate_source_linkage_invalid"
    if rerun_summary.get("status") != "passed":
        return "failed", ROUTE_MISSING_SOURCE, "stage23_0_rerun_failed"
    if bool(obstacle_audit.get("only_proxy_sources_available", False)):
        return "passed", ROUTE_PROXY_REVIEW, "only_proxy_sources_available"
    if bool(rerun_summary.get("obstacle_occlusion_material_to_coverage", False)):
        return "passed", ROUTE_STAGE23_1, "obstacle_occlusion_material_after_source_materialization"
    return "passed", ROUTE_AUDIT_ONLY, "obstacle_occlusion_not_material_after_source_materialization"


def _blocking_reasons(obstacle_audit: dict[str, Any], linkage_audit: dict[str, Any], rerun_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(obstacle_audit.get("source_count", 0)) <= 0:
        reasons.append("obstacle_source_absent")
    if int(linkage_audit.get("candidate_obstacle_source_missing_count", 0)) > 0:
        reasons.append("candidate_obstacle_source_missing")
    if int(linkage_audit.get("candidate_obstacle_source_hash_mismatch_count", 0)) > 0:
        reasons.append("candidate_obstacle_source_hash_mismatch")
    if rerun_summary.get("status") == "failed":
        reasons.append("stage23_0_rerun_failed")
    return reasons


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
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unexpected config schema_version: {payload.get('schema_version')}")
    payload.setdefault("run_high_fidelity_smoke", True)
    payload.setdefault("high_fidelity_config", DEFAULT_HIGH_FIDELITY_CONFIG)
    payload.setdefault("high_fidelity_output_subdir", "hf")
    payload.setdefault("required_scenario_count", 2)
    payload.setdefault("rollout_steps", 4)
    payload.setdefault("theta_bin_count", 8)
    payload.setdefault("theta_step_deg", 45)
    payload.setdefault("sensor_fov_deg", 90.0)
    payload.setdefault("sensor_range_cells", 3)
    payload.setdefault("max_audit_rows", 5000)
    payload.setdefault("obstacle_occlusion_enabled", False)
    payload.setdefault("no_go_blocks_los", False)
    payload.setdefault("stage23_0a_authorized", False)
    payload.setdefault("runs_new_ppo_update", False)
    payload.setdefault("publishes_checkpoint", False)
    payload.setdefault("replaces_default_policy", False)
    payload.setdefault("connects_real_executor", False)
    payload.setdefault("starts_online_canary", False)
    payload.setdefault("canary_traffic_fraction", 0.0)
    return apply_stage23_platform_defaults(payload, repo_root=repo_root)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage23.0A Materialize Obstacle Sources For Endpoint Theta LOS",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- obstacle_source_count: `{summary['obstacle_source_count']}`",
            f"- candidate_obstacle_source_missing_count: `{summary['candidate_obstacle_source_missing_count']}`",
            f"- candidate_obstacle_source_hash_mismatch_count: `{summary['candidate_obstacle_source_hash_mismatch_count']}`",
            f"- stage23_0_los_audit_row_count: `{summary['stage23_0_los_audit_row_count']}`",
            "",
            "Stage23.0A 只物化 endpoint theta LOS 所需障碍源并重跑 Stage23.0；不训练、不发布、不替换默认策略。",
            "",
        ]
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
