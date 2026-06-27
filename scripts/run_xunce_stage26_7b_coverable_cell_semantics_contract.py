from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_high_fidelity_exploration_coverage_comparison as hf
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_high_fidelity_exploration_coverage_comparison as hf


STAGE_ID = "xunce-stage26-7b-coverable-cell-semantics-contract"
CONFIG_SCHEMA_VERSION = "xunce-stage26-7b-coverable-cell-semantics-contract-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-7b-summary/v1"
SEMANTICS_SCHEMA_VERSION = "xunce-stage26-7b-coverable-cell-semantics-audit/v1"
DENOMINATOR_SCHEMA_VERSION = "xunce-stage26-7b-coverage-denominator-comparison/v1"
HAZARD_SCHEMA_VERSION = "xunce-stage26-7b-hazard-observation-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-7b-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-7b-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_7b_coverable_cell_semantics_contract_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_7b_coverable_cell_semantics_contract_v1"
)
DEFAULT_STAGE26_7_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair_v1"
)
DEFAULT_STAGE26_0_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract_v1"
)

SUMMARY_FILE = "xunce-stage26-7b-summary.json"
SEMANTICS_FILE = "xunce-stage26-7b-coverable-cell-semantics-audit.json"
DENOMINATOR_FILE = "xunce-stage26-7b-coverage-denominator-comparison.json"
HAZARD_FILE = "xunce-stage26-7b-hazard-observation-audit.json"
ROUTING_FILE = "xunce-stage26-7b-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-7b-report.md"
MANIFEST_FILE = "xunce-stage26-7b-manifest.json"

ROUTE_INPUTS = "rerun_stage26_7b_required_inputs"
ROUTE_SEMANTICS = "repair_stage26_7b_cell_semantics_contract"
ROUTE_MAIN_DENOMINATOR = "repair_stage26_7b_main_coverable_denominator"
ROUTE_SYNTHETIC_SEMANTICS = "repair_stage26_7b_synthetic_source_semantics"
ROUTE_HASH = "repair_stage26_7b_denominator_hash_contract"
ROUTE_NEXT = "rerun_stage26_7_path_efficiency_with_main_coverable_coverage"
ROUTE_BOUNDARY = "resolve_stage26_7b_boundary_rejections"

BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Stage26.7B coverable cell semantics.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_7b_coverable_cell_semantics_contract(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_7b_coverable_cell_semantics_contract(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_7_root = Path(config["stage26_7_root"])
    stage26_7_summary = _read_json_if_exists(stage26_7_root / "xunce-stage26-7-summary.json")
    eval_comparison = _read_json_if_exists(stage26_7_root / "xunce-stage26-7-post-update-eval-comparison.json")
    best_combo_root = _best_combo_root(stage26_7_root, eval_comparison, config)

    input_rejections = _input_rejections(stage26_7_summary, best_combo_root)
    boundary_rejections = _boundary_rejections(config)
    semantics_audit = _semantic_audit(best_combo_root, config) if not input_rejections else _empty_semantics_audit()
    denominator_comparison = _denominator_comparison(best_combo_root, semantics_audit, eval_comparison)
    hazard_audit = _hazard_observation_audit(semantics_audit, best_combo_root)

    status, route, reasons = _route(
        input_rejections=input_rejections,
        boundary_rejections=boundary_rejections,
        semantics_audit=semantics_audit,
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "blocking_reason_codes": reasons,
        "stage26_7_root": str(stage26_7_root),
        "stage26_7_status": stage26_7_summary.get("status"),
        "stage26_7_next_required_change": stage26_7_summary.get("next_required_change"),
        "best_combo_root": str(best_combo_root) if best_combo_root is not None else None,
        "coverage_denominator_mode": "main_coverable_cells",
        "coverage_denominator_source": "main_coverable_cells/v1",
        "sidecar_count": semantics_audit["sidecar_count"],
        "main_coverable_denominator_cells_total": semantics_audit["main_coverable_denominator_cells_total"],
        "hazard_observable_denominator_cells_total": semantics_audit["hazard_observable_denominator_cells_total"],
        "hard_obstacle_in_main_denominator_count": semantics_audit["hard_obstacle_in_main_denominator_count"],
        "los_only_passable_not_main_count": semantics_audit["los_only_passable_not_main_count"],
        "physical_obstacle_cells_written_count": semantics_audit["physical_obstacle_cells_written_count"],
        "physical_obstacle_payload_count": semantics_audit["physical_obstacle_payload_count"],
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "max_traversable_slope_deg": 30.0,
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
        "blocking_reason_codes": reasons,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "summary": str(output_root / SUMMARY_FILE),
        "coverable_cell_semantics_audit": str(output_root / SEMANTICS_FILE),
        "coverage_denominator_comparison": str(output_root / DENOMINATOR_FILE),
        "hazard_observation_audit": str(output_root / HAZARD_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / SEMANTICS_FILE, semantics_audit)
    _write_json(output_root / DENOMINATOR_FILE, denominator_comparison)
    _write_json(output_root / HAZARD_FILE, hazard_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    _write_report(output_root / REPORT_FILE, summary, semantics_audit, denominator_comparison, hazard_audit)
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    if payload.get("coverage_denominator_mode", "main_coverable_cells") != "main_coverable_cells":
        raise ValueError("coverage_denominator_mode must be main_coverable_cells")
    if payload.get("coverage_denominator_source", "main_coverable_cells/v1") != "main_coverable_cells/v1":
        raise ValueError("coverage_denominator_source must be main_coverable_cells/v1")
    config = dict(payload)
    config["stage26_7_root"] = str(_resolve_path(Path(payload.get("stage26_7_root", DEFAULT_STAGE26_7_ROOT)), repo_root))
    config["stage26_0_root"] = str(_resolve_path(Path(payload.get("stage26_0_root", DEFAULT_STAGE26_0_ROOT)), repo_root))
    config["best_combo_work_dir"] = str(payload.get("best_combo_work_dir", "c2"))
    config["derive_slope_blocked_cells_from_sidecar_dem"] = bool(
        payload.get("derive_slope_blocked_cells_from_sidecar_dem", False)
    )
    config["max_traversable_slope_deg"] = float(payload.get("max_traversable_slope_deg", 30.0))
    for field in BOUNDARY_FIELDS:
        config[field] = bool(payload.get(field, False))
    config["canary_traffic_fraction"] = float(payload.get("canary_traffic_fraction", 0.0))
    return config


def _input_rejections(stage26_7_summary: dict[str, Any], best_combo_root: Path | None) -> list[str]:
    rejections: list[str] = []
    if stage26_7_summary.get("stage_id") != "xunce-stage26-7-synthetic-credit-assignment-path-efficiency-repair":
        rejections.append("stage26_7_summary_missing_or_wrong_stage")
    if best_combo_root is None or not best_combo_root.is_dir():
        rejections.append("best_combo_root_missing")
    return rejections


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    rejections = [field for field in BOUNDARY_FIELDS if bool(config.get(field))]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        rejections.append("canary_traffic_fraction")
    return rejections


def _best_combo_root(stage26_7_root: Path, eval_comparison: dict[str, Any], config: dict[str, Any]) -> Path | None:
    combo_root = eval_comparison.get("best_combo", {}).get("combo_root") if isinstance(eval_comparison.get("best_combo"), dict) else None
    if isinstance(combo_root, str) and combo_root.strip():
        return Path(combo_root)
    work_dir = str(config.get("best_combo_work_dir", "c2"))
    return stage26_7_root / work_dir


def _semantic_audit(best_combo_root: Path | None, config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if best_combo_root is None:
        return _empty_semantics_audit()
    for sidecar_path in sorted((best_combo_root / "s26_1" / "src" / "sc").glob("*.sidecar.json")):
        semantics = hf._sidecar_coverable_cell_semantics(
            sidecar_path,
            derive_slope_blocked=bool(config.get("derive_slope_blocked_cells_from_sidecar_dem", False)),
            max_traversable_slope_deg=float(config.get("max_traversable_slope_deg", 30.0)),
        )
        payload = _read_json_if_exists(sidecar_path)
        row = _public_semantic_row(sidecar_path, semantics, payload)
        rows.append(row)
    hard_overlap = sum(int(row["main_coverable_hard_obstacle_overlap_count"]) for row in rows)
    los_only_not_main = sum(int(row["los_only_passable_not_main_count"]) for row in rows)
    physical_pollution = sum(1 for row in rows if bool(row["physical_obstacle_cells_written"]))
    physical_payload = sum(1 for row in rows if bool(row["physical_obstacle_payload_present"]))
    missing_hash = sum(1 for row in rows if not row.get("coverable_cell_semantics_hash"))
    return {
        "schema_version": SEMANTICS_SCHEMA_VERSION,
        "sidecar_count": len(rows),
        "rows": rows,
        "main_coverable_denominator_cells_total": sum(int(row["main_coverable_denominator_cells"]) for row in rows),
        "passable_denominator_cells_total": sum(int(row["passable_denominator_cells"]) for row in rows),
        "raw_roi_denominator_cells_total": sum(int(row["raw_roi_denominator_cells"]) for row in rows),
        "hazard_observable_denominator_cells_total": sum(int(row["hazard_observable_denominator_cells"]) for row in rows),
        "hard_obstacle_in_main_denominator_count": hard_overlap,
        "los_only_passable_not_main_count": los_only_not_main,
        "physical_obstacle_cells_written_count": physical_pollution,
        "physical_obstacle_payload_count": physical_payload,
        "missing_semantics_hash_count": missing_hash,
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
    }


def _empty_semantics_audit() -> dict[str, Any]:
    return {
        "schema_version": SEMANTICS_SCHEMA_VERSION,
        "sidecar_count": 0,
        "rows": [],
        "main_coverable_denominator_cells_total": 0,
        "passable_denominator_cells_total": 0,
        "raw_roi_denominator_cells_total": 0,
        "hazard_observable_denominator_cells_total": 0,
        "hard_obstacle_in_main_denominator_count": 0,
        "los_only_passable_not_main_count": 0,
        "physical_obstacle_cells_written_count": 0,
        "physical_obstacle_payload_count": 0,
        "missing_semantics_hash_count": 0,
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
    }


def _public_semantic_row(sidecar_path: Path, semantics: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    los_not_main = int(semantics.get("synthetic_los_only_blocker_passable_count") or 0) - int(
        semantics.get("synthetic_los_only_blocker_main_coverable_count") or 0
    )
    return {
        "sidecar_path": str(sidecar_path),
        "coverable_cell_semantics_hash": semantics.get("coverable_cell_semantics_hash"),
        "raw_roi_denominator_cells": int(semantics.get("raw_roi_denominator_cells") or 0),
        "passable_denominator_cells": int(semantics.get("passable_denominator_cells") or 0),
        "main_coverable_denominator_cells": int(semantics.get("main_coverable_denominator_cells") or 0),
        "hazard_observable_denominator_cells": int(semantics.get("hazard_observable_denominator_cells") or 0),
        "traversable_cell_count": int(semantics.get("traversable_cell_count") or 0),
        "hard_obstacle_cell_count": int(semantics.get("hard_obstacle_cell_count") or 0),
        "main_coverable_hard_obstacle_overlap_count": int(
            semantics.get("main_coverable_hard_obstacle_overlap_count") or 0
        ),
        "los_blocker_cell_count": int(semantics.get("los_blocker_cell_count") or 0),
        "synthetic_los_only_blocker_cell_count": int(semantics.get("synthetic_los_only_blocker_cell_count") or 0),
        "synthetic_los_only_blocker_passable_count": int(
            semantics.get("synthetic_los_only_blocker_passable_count") or 0
        ),
        "synthetic_los_only_blocker_main_coverable_count": int(
            semantics.get("synthetic_los_only_blocker_main_coverable_count") or 0
        ),
        "los_only_passable_not_main_count": max(0, los_not_main),
        "physical_obstacle_cell_count": _cell_count(payload.get("physical_obstacle_cells")),
        "physical_obstacle_payload_present": _cell_count(payload.get("physical_obstacle_cells")) > 0,
        "physical_obstacle_cells_written": bool(payload.get("physical_obstacle_cells_written", False)),
        "synthetic_source_kind": payload.get("synthetic_source_kind"),
    }


def _denominator_comparison(
    best_combo_root: Path | None,
    semantics_audit: dict[str, Any],
    eval_comparison: dict[str, Any],
) -> dict[str, Any]:
    best_combo = eval_comparison.get("best_combo", {}) if isinstance(eval_comparison.get("best_combo"), dict) else {}
    pre_rows = _episode_rows(best_combo_root, "pre")
    post_rows = _episode_rows(best_combo_root, "post")
    return {
        "schema_version": DENOMINATOR_SCHEMA_VERSION,
        "coverage_denominator_source": "main_coverable_cells/v1",
        "raw_roi_denominator_cells_total": semantics_audit["raw_roi_denominator_cells_total"],
        "passable_denominator_cells_total": semantics_audit["passable_denominator_cells_total"],
        "main_coverable_denominator_cells_total": semantics_audit["main_coverable_denominator_cells_total"],
        "main_vs_passable_denominator_delta_total": semantics_audit["main_coverable_denominator_cells_total"]
        - semantics_audit["passable_denominator_cells_total"],
        "stage26_7_best_combo_id": best_combo.get("combo_id"),
        "legacy_final_coverage_delta": best_combo.get("final_coverage_delta"),
        "legacy_coverage_auc_delta": best_combo.get("coverage_auc_delta"),
        "legacy_coverage_per_100m_delta": best_combo.get("coverage_per_100m_delta"),
        "legacy_hybrid_astar_path_cost_delta": best_combo.get("hybrid_astar_path_cost_delta"),
        "pre_episode_count": len(pre_rows),
        "post_episode_count": len(post_rows),
        "pre_existing_main_coverage_rate_mean": _mean_existing_field(pre_rows, "main_coverage_rate"),
        "post_existing_main_coverage_rate_mean": _mean_existing_field(post_rows, "main_coverage_rate"),
        "requires_stage26_7_rerun_for_exact_main_coverable_auc": (
            _mean_existing_field(pre_rows, "main_coverage_rate") is None
            or _mean_existing_field(post_rows, "main_coverage_rate") is None
        ),
    }


def _hazard_observation_audit(semantics_audit: dict[str, Any], best_combo_root: Path | None) -> dict[str, Any]:
    pre_rows = _episode_rows(best_combo_root, "pre")
    post_rows = _episode_rows(best_combo_root, "post")
    return {
        "schema_version": HAZARD_SCHEMA_VERSION,
        "hazard_observable_denominator_cells_total": semantics_audit["hazard_observable_denominator_cells_total"],
        "pre_hazard_observation_rate_mean": _mean_existing_field(pre_rows, "hazard_observation_rate"),
        "post_hazard_observation_rate_mean": _mean_existing_field(post_rows, "hazard_observation_rate"),
        "requires_stage26_7_rerun_for_exact_hazard_observation_rate": (
            _mean_existing_field(pre_rows, "hazard_observation_rate") is None
            or _mean_existing_field(post_rows, "hazard_observation_rate") is None
        ),
        "hazard_semantics": {
            "hard_or_risk_hazards_are_observable_targets": True,
            "hard_or_blocked_hazards_are_excluded_from_main_coverable_denominator": True,
            "synthetic_los_only_blockers_are_not_automatically_excluded": True,
        },
    }


def _route(
    *,
    input_rejections: list[str],
    boundary_rejections: list[str],
    semantics_audit: dict[str, Any],
) -> tuple[str, str, list[str]]:
    if boundary_rejections:
        return "failed", ROUTE_BOUNDARY, boundary_rejections
    if input_rejections:
        return "failed", ROUTE_INPUTS, input_rejections
    if int(semantics_audit["sidecar_count"]) <= 0:
        return "failed", ROUTE_INPUTS, ["sidecar_semantics_missing"]
    if int(semantics_audit["physical_obstacle_cells_written_count"]) > 0:
        return "failed", ROUTE_SYNTHETIC_SEMANTICS, ["synthetic_physical_obstacle_pollution"]
    if int(semantics_audit["physical_obstacle_payload_count"]) > 0:
        return "failed", ROUTE_SYNTHETIC_SEMANTICS, ["synthetic_physical_obstacle_payload_present"]
    if int(semantics_audit["hard_obstacle_in_main_denominator_count"]) > 0:
        return "failed", ROUTE_MAIN_DENOMINATOR, ["hard_obstacle_in_main_coverable_denominator"]
    if int(semantics_audit["los_only_passable_not_main_count"]) > 0:
        return "failed", ROUTE_SEMANTICS, ["los_only_blocker_wrongly_removed_from_main_coverable"]
    if int(semantics_audit["missing_semantics_hash_count"]) > 0:
        return "failed", ROUTE_HASH, ["coverable_cell_semantics_hash_missing"]
    return "passed", ROUTE_NEXT, []


def _episode_rows(best_combo_root: Path | None, split: str) -> list[dict[str, Any]]:
    if best_combo_root is None:
        return []
    path = best_combo_root / "s26_3" / split / "xunce-exploration-coverage-episodes.jsonl"
    return _read_jsonl_if_exists(path)


def _mean_existing_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _cell_count(value: Any) -> int:
    if not isinstance(value, list):
        return 0
    count = 0
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            count += 1
    return count


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    summary: dict[str, Any],
    semantics_audit: dict[str, Any],
    denominator_comparison: dict[str, Any],
    hazard_audit: dict[str, Any],
) -> None:
    lines = [
        "# Stage26.7B Coverable Cell Semantics Contract Report",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- sidecar_count: `{summary['sidecar_count']}`",
        f"- main_coverable_denominator_cells_total: `{summary['main_coverable_denominator_cells_total']}`",
        f"- hazard_observable_denominator_cells_total: `{summary['hazard_observable_denominator_cells_total']}`",
        "",
        "Stage26.7B only recalculates coverage denominator semantics. It does not run PPO, change reward, change network, regenerate synthetic terrain, replace default A*, publish checkpoints, connect executors, or start canary traffic.",
        "",
        "## Denominator Comparison",
        "",
        "```json",
        json.dumps(denominator_comparison, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Hazard Observation",
        "",
        "```json",
        json.dumps(hazard_audit, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Semantic Audit",
        "",
        "```json",
        json.dumps(semantics_audit, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
