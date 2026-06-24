from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los import (
    OBSTACLE_SOURCE_AUDIT_FILE as STAGE23_0A_OBSTACLE_SOURCE_AUDIT_FILE,
    SUMMARY_FILE as STAGE23_0A_SUMMARY_FILE,
    run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los,
)


CONFIG_SCHEMA_VERSION = "xunce-stage23-0b-map-obstacle-source-export-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage23-0b-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage23-0b-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage23-0b-manifest/v1"
STAGE_ID = "xunce-stage23-0b-map-obstacle-source-export"

DEFAULT_CONFIG = "configs/xunce_stage23_0b_map_obstacle_source_export_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage23_endpoint_obstacle_aware_theta_sensor_coverage/"
    "outputs/path_feedback_batch_xunce_stage23_0b_map_obstacle_source_export_v1"
)

SUMMARY_FILE = "xunce-stage23-0b-summary.json"
SIDECAR_AUDIT_FILE = "xunce-stage23-0b-sidecar-obstacle-source-audit.json"
RERUN_STAGE23_0A_SUMMARY_FILE = "xunce-stage23-0b-rerun-stage23-0a-summary.json"
ROUTING_FILE = "xunce-stage23-0b-next-stage-routing.json"
REPORT_FILE = "xunce-stage23-0b-report.md"
MANIFEST_FILE = "xunce-stage23-0b-manifest.json"
PROXY_OBSTACLE_SOURCE_KINDS = {
    "blocked_as_obstacle_proxy",
    "slope_blocked_as_obstacle_proxy",
    "no_go_as_obstacle_proxy",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage23.0B map obstacle source export repair.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage23_0b_map_obstacle_source_export(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(
        json.dumps(
            {"status": summary["status"], "next_required_change": summary["next_required_change"]},
            ensure_ascii=False,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage23_0b_map_obstacle_source_export(
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
    stage23_0a_summary = _read_stage23_0a_summary(stage23_0a_root)
    sidecar_audit = _sidecar_source_audit(stage23_0a_root, stage23_0a_summary)
    sidecar_export_audit = _sidecar_export_audit_from_stage23_0a_summary(stage23_0a_summary)
    sidecar_audit["sidecar_export_audit"] = sidecar_export_audit
    status, route, route_reason, blockers = _route(boundary_reasons, stage23_0a_summary, sidecar_audit)

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
        "stage23_0a_obstacle_source_count": sidecar_audit["source_count"],
        "stage23_0a_obstacle_source_kind_counts": sidecar_audit["obstacle_source_kind_counts"],
        "only_blocked_proxy_sources_available": sidecar_audit["only_blocked_proxy_sources_available"],
        "only_proxy_sources_available": sidecar_audit["only_proxy_sources_available"],
        "physical_obstacle_source_available": sidecar_audit["physical_obstacle_source_available"],
        "stage23_0a_candidate_obstacle_source_missing_count": int(
            stage23_0a_summary.get("candidate_obstacle_source_missing_count", 0)
        ),
        "stage23_0a_candidate_obstacle_source_hash_mismatch_count": int(
            stage23_0a_summary.get("candidate_obstacle_source_hash_mismatch_count", 0)
        ),
        "stage23_0_los_audit_row_count": int(stage23_0a_summary.get("stage23_0_los_audit_row_count", 0)),
        "stage23_0_obstacle_occlusion_material_to_coverage": bool(
            stage23_0a_summary.get("stage23_0_obstacle_occlusion_material_to_coverage", False)
        ),
        "source_sidecar_count": sidecar_export_audit["sidecar_count"],
        "source_passable_mask_false_cell_count_total": sidecar_export_audit["passable_mask_false_cell_count_total"],
        "source_sidecar_blocked_cells_cell_count_total": sidecar_export_audit["sidecar_blocked_cells_cell_count_total"],
        "source_sidecar_slope_blocked_cells_cell_count_total": sidecar_export_audit[
            "sidecar_slope_blocked_cells_cell_count_total"
        ],
        "stage23_0b_authorized": False,
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
        "stage23_0b_authorized": False,
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
        "sidecar_obstacle_source_audit": str(output_root / SIDECAR_AUDIT_FILE),
        "rerun_stage23_0a_summary": str(output_root / RERUN_STAGE23_0A_SUMMARY_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / SIDECAR_AUDIT_FILE, sidecar_audit)
    _write_json(output_root / RERUN_STAGE23_0A_SUMMARY_FILE, stage23_0a_summary)
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
    default_stage23_0a_config = "configs/xunce_stage23_0a_materialize_obstacle_sources_for_theta_los_v1.json"
    stage23_0a_config = _resolve_path(Path(str(config.get("stage23_0a_config", default_stage23_0a_config))), repo_root)
    stage23_0a_root = output_root / str(config.get("stage23_0a_output_subdir", "s23_0a"))
    run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los(
        config_path=stage23_0a_config,
        output_root=stage23_0a_root,
        repo_root=repo_root,
    )
    return stage23_0a_root


def _read_stage23_0a_summary(stage23_0a_root: Path) -> dict[str, Any]:
    path = stage23_0a_root / STAGE23_0A_SUMMARY_FILE
    if not path.exists():
        return {"status": "failed", "next_required_change": "rerun_stage23_0a_required_inputs"}
    return _read_json(path)


def _sidecar_source_audit(stage23_0a_root: Path, stage23_0a_summary: dict[str, Any]) -> dict[str, Any]:
    path = stage23_0a_root / STAGE23_0A_OBSTACLE_SOURCE_AUDIT_FILE
    payload = _read_json(path) if path.exists() else {}
    kind_counts = (
        payload.get("obstacle_source_kind_counts")
        or stage23_0a_summary.get("obstacle_source_kind_counts")
        or {}
    )
    if not isinstance(kind_counts, dict):
        kind_counts = {}
    source_count = int(payload.get("source_count", stage23_0a_summary.get("obstacle_source_count", 0)) or 0)
    kind_names = {str(key) for key, value in kind_counts.items() if int(value or 0) > 0}
    computed_only_proxy = source_count > 0 and bool(kind_names) and kind_names.issubset(PROXY_OBSTACLE_SOURCE_KINDS)
    only_blocked_proxy = bool(
        payload.get(
            "only_blocked_proxy_sources_available",
            stage23_0a_summary.get("only_blocked_proxy_sources_available", False),
        )
    )
    only_proxy = bool(
        payload.get(
            "only_proxy_sources_available",
            stage23_0a_summary.get("only_proxy_sources_available", computed_only_proxy),
        )
    )
    physical_available = int(kind_counts.get("physical_obstacle_cells", 0) or 0) > 0
    return {
        "schema_version": "xunce-stage23-0b-sidecar-obstacle-source-audit/v1",
        "stage23_0a_obstacle_source_audit": str(path) if path.exists() else None,
        "source_count": source_count,
        "obstacle_source_kind_counts": kind_counts,
        "only_blocked_proxy_sources_available": only_blocked_proxy,
        "only_proxy_sources_available": only_proxy,
        "physical_obstacle_source_available": physical_available,
        "candidate_obstacle_source_missing_count": int(
            stage23_0a_summary.get("candidate_obstacle_source_missing_count", 0) or 0
        ),
        "candidate_obstacle_source_hash_mismatch_count": int(
            stage23_0a_summary.get("candidate_obstacle_source_hash_mismatch_count", 0) or 0
        ),
        "stage23_0_los_audit_row_count": int(stage23_0a_summary.get("stage23_0_los_audit_row_count", 0) or 0),
    }


def _sidecar_export_audit_from_stage23_0a_summary(stage23_0a_summary: dict[str, Any]) -> dict[str, Any]:
    high_fidelity_root_value = stage23_0a_summary.get("high_fidelity_root")
    if not high_fidelity_root_value:
        return _empty_sidecar_export_audit(None)
    manifest_path = Path(str(high_fidelity_root_value)) / "xunce-exploration-coverage-comparison-manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    source_root = (
        manifest.get("normalized_config", {}).get("source_roi_expansion_root")
        if isinstance(manifest.get("normalized_config"), dict)
        else None
    )
    if not source_root:
        return _empty_sidecar_export_audit(None)
    return _sidecar_export_audit_from_source_root(Path(str(source_root)))


def _sidecar_export_audit_from_source_root(source_root: Path) -> dict[str, Any]:
    if not source_root.exists():
        return _empty_sidecar_export_audit(str(source_root))
    slices_path = source_root / "xunce-high-fidelity-real-map-slices.jsonl"
    if not slices_path.exists():
        slices_path = source_root / "quasi-real-map-slices.jsonl"
    sidecar_count = 0
    sidecar_with_blocked = 0
    sidecar_with_passable_false = 0
    blocked_cells_total = 0
    slope_blocked_cells_total = 0
    passable_false_total = 0
    blocked_kind_counts: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    if not slices_path.exists():
        return _empty_sidecar_export_audit(str(source_root))
    for line in slices_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sidecar_path = Path(str(row.get("sidecar", "")))
        if not sidecar_path.is_absolute():
            sidecar_path = source_root / sidecar_path
        if not sidecar_path.exists():
            continue
        payload = _read_json(sidecar_path)
        sidecar_count += 1
        false_count = _passable_mask_false_count(payload.get("passable_mask"))
        blocked_cells = payload.get("blocked_cells")
        blocked_count = len(blocked_cells) if isinstance(blocked_cells, list) else 0
        slope_blocked_cells = payload.get("slope_blocked_cells")
        slope_blocked_count = len(slope_blocked_cells) if isinstance(slope_blocked_cells, list) else 0
        blocked_kind = str(payload.get("blocked_source_kind") or "missing")
        if false_count > 0:
            sidecar_with_passable_false += 1
        if blocked_count > 0:
            sidecar_with_blocked += 1
        passable_false_total += false_count
        blocked_cells_total += blocked_count
        slope_blocked_cells_total += slope_blocked_count
        blocked_kind_counts[blocked_kind] = blocked_kind_counts.get(blocked_kind, 0) + 1
        if (false_count > 0 or blocked_count > 0) and len(examples) < 5:
            examples.append(
                {
                    "scenario_id": row.get("scenario_id"),
                    "sidecar": str(sidecar_path),
                    "passable_mask_false_cell_count": false_count,
                    "blocked_cells_cell_count": blocked_count,
                    "slope_blocked_cells_cell_count": slope_blocked_count,
                    "blocked_source_kind": blocked_kind,
                }
            )
    return {
        "schema_version": "xunce-stage23-0b-sidecar-export-audit/v1",
        "source_roi_expansion_root": str(source_root),
        "slices_path": str(slices_path),
        "sidecar_count": sidecar_count,
        "sidecar_with_passable_mask_false_count": sidecar_with_passable_false,
        "sidecar_with_blocked_cells_count": sidecar_with_blocked,
        "passable_mask_false_cell_count_total": passable_false_total,
        "sidecar_blocked_cells_cell_count_total": blocked_cells_total,
        "sidecar_slope_blocked_cells_cell_count_total": slope_blocked_cells_total,
        "blocked_source_kind_counts": blocked_kind_counts,
        "example_sidecars_with_sources": examples,
    }


def _passable_mask_false_count(mask: Any) -> int:
    if not isinstance(mask, list):
        return 0
    count = 0
    for row in mask:
        if not isinstance(row, list):
            continue
        count += sum(1 for value in row if value is False)
    return count


def _empty_sidecar_export_audit(source_root: str | None) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage23-0b-sidecar-export-audit/v1",
        "source_roi_expansion_root": source_root,
        "slices_path": None,
        "sidecar_count": 0,
        "sidecar_with_passable_mask_false_count": 0,
        "sidecar_with_blocked_cells_count": 0,
        "passable_mask_false_cell_count_total": 0,
        "sidecar_blocked_cells_cell_count_total": 0,
        "sidecar_slope_blocked_cells_cell_count_total": 0,
        "blocked_source_kind_counts": {},
        "example_sidecars_with_sources": [],
    }


def _route(
    boundary_reasons: list[str],
    stage23_0a_summary: dict[str, Any],
    sidecar_audit: dict[str, Any],
) -> tuple[str, str, str, list[str]]:
    if boundary_reasons:
        return "failed", "resolve_stage23_0b_boundary_rejections", "boundary_rejection", sorted(boundary_reasons)
    source_count = int(sidecar_audit.get("source_count", 0))
    missing = int(sidecar_audit.get("candidate_obstacle_source_missing_count", 0))
    mismatch = int(sidecar_audit.get("candidate_obstacle_source_hash_mismatch_count", 0))
    los_rows = int(sidecar_audit.get("stage23_0_los_audit_row_count", 0))
    stage23_0a_status = str(stage23_0a_summary.get("status") or "")
    stage23_0_status = str(stage23_0a_summary.get("stage23_0_status") or "")
    material = bool(stage23_0a_summary.get("stage23_0_obstacle_occlusion_material_to_coverage", False))
    if mismatch > 0:
        return (
            "failed",
            "repair_stage23_obstacle_source_lineage",
            "candidate_source_hash_mismatch",
            ["candidate_source_hash_mismatch"],
        )
    if source_count > 0 and missing > 0:
        return (
            "failed",
            "repair_stage23_obstacle_source_lineage",
            "candidate_source_missing_with_source_artifact",
            ["candidate_obstacle_source_missing"],
        )
    blockers: list[str] = []
    if source_count <= 0 or missing > 0 or los_rows <= 0:
        if source_count <= 0:
            blockers.append("stage23_0a_still_missing_obstacle_sources")
        if missing > 0:
            blockers.append("candidate_obstacle_source_missing")
        if los_rows <= 0:
            blockers.append("stage23_0_los_audit_missing")
        return (
            "failed",
            "continue_stage23_map_obstacle_source_export_repair",
            "stage23_0a_still_missing_usable_sources",
            sorted(set(blockers)),
        )
    if stage23_0a_status != "passed" or stage23_0_status != "passed":
        return (
            "failed",
            "continue_stage23_map_obstacle_source_export_repair",
            "stage23_0a_or_stage23_0_not_passed",
            ["stage23_0a_or_stage23_0_not_passed"],
        )
    if sidecar_audit.get("physical_obstacle_source_available"):
        if not material:
            return (
                "passed",
                "document_obstacle_occlusion_audit_only",
                "physical_source_not_material_to_coverage",
                [],
            )
        return (
            "passed",
            "implement_stage23_1_obstacle_aware_theta_reward_contract",
            "physical_obstacle_source_material_to_coverage",
            [],
        )
    if sidecar_audit.get("only_proxy_sources_available"):
        return (
            "passed",
            "review_blocked_as_obstacle_proxy_semantics",
            "only_proxy_sources_available",
            [],
        )
    return (
        "failed",
        "repair_stage23_obstacle_source_lineage",
        "ambiguous_obstacle_source_semantics",
        ["ambiguous_obstacle_source_semantics"],
    )


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in (
        "stage23_0b_authorized",
        "runs_new_ppo_update",
        "publishes_checkpoint",
        "replaces_default_policy",
        "connects_real_executor",
        "starts_online_canary",
    ):
        if bool(config.get(key, False)):
            reasons.append(f"{key}_must_be_false")
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction_must_be_zero")
    return reasons


def _load_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    path = _resolve_path(config_path, repo_root)
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    return payload


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage23.0B Map Obstacle Source Export",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- stage23_0a_obstacle_source_count: `{summary['stage23_0a_obstacle_source_count']}`",
            "- stage23_0a_candidate_obstacle_source_missing_count: "
            f"`{summary['stage23_0a_candidate_obstacle_source_missing_count']}`",
            f"- stage23_0_los_audit_row_count: `{summary['stage23_0_los_audit_row_count']}`",
            "- stage23_0_obstacle_occlusion_material_to_coverage: "
            f"`{summary['stage23_0_obstacle_occlusion_material_to_coverage']}`",
            f"- source_sidecar_count: `{summary['source_sidecar_count']}`",
            "- source_passable_mask_false_cell_count_total: "
            f"`{summary['source_passable_mask_false_cell_count_total']}`",
            "- source_sidecar_blocked_cells_cell_count_total: "
            f"`{summary['source_sidecar_blocked_cells_cell_count_total']}`",
            "- source_sidecar_slope_blocked_cells_cell_count_total: "
            f"`{summary['source_sidecar_slope_blocked_cells_cell_count_total']}`",
            f"- physical_obstacle_source_available: `{summary['physical_obstacle_source_available']}`",
            f"- only_blocked_proxy_sources_available: `{summary['only_blocked_proxy_sources_available']}`",
            f"- only_proxy_sources_available: `{summary['only_proxy_sources_available']}`",
            "",
            "Stage23.0B 只修复/审计 endpoint theta LOS 的地图障碍源导出；不训练、不发布、不替换默认策略。",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
