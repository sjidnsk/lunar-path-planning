from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


try:
    from model_explorer.data.geotiff import GeoTiffDecodeUnavailable, read_geotiff_window
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model-explorer" / "src"))
    from model_explorer.data.geotiff import GeoTiffDecodeUnavailable, read_geotiff_window

try:
    import run_xunce_stage23_2a_high_resolution_terrain_data_prepare as stage23_2a
    import run_xunce_stage23_2b_platform_geometry_sensor_contract_alignment as stage23_2b
    import run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract as stage23_2
    import run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke as stage23_3
    import run_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke as stage23_4
    import run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as stage23_5
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage23_2a_high_resolution_terrain_data_prepare as stage23_2a
    import scripts.run_xunce_stage23_2b_platform_geometry_sensor_contract_alignment as stage23_2b
    import scripts.run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract as stage23_2
    import scripts.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke as stage23_3
    import scripts.run_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke as stage23_4
    import scripts.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as stage23_5


STAGE_ID = "xunce-stage23-5a-repair-required-inputs-high-res-roi-coverage"
CONFIG_SCHEMA_VERSION = "xunce-stage23-5a-repair-required-inputs-high-res-roi-coverage-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage23-5a-summary/v1"
ROI_AUDIT_SCHEMA_VERSION = "xunce-stage23-5a-roi-window-selection-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage23-5a-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage23-5a-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage23_slope_obstacle_aware_theta_reward/"
    "outputs/path_feedback_batch_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage_v1"
)

SUMMARY_FILE = "xunce-stage23-5a-summary.json"
ROI_AUDIT_FILE = "xunce-stage23-5a-roi-window-selection-audit.json"
RERUN_23_2B_FILE = "xunce-stage23-5a-rerun-stage23-2b-summary.json"
RERUN_23_2_FILE = "xunce-stage23-5a-rerun-stage23-2-summary.json"
RERUN_23_3_FILE = "xunce-stage23-5a-rerun-stage23-3-summary.json"
RERUN_23_4_FILE = "xunce-stage23-5a-rerun-stage23-4-summary.json"
RERUN_23_5_FILE = "xunce-stage23-5a-rerun-stage23-5-summary.json"
RERUN_STAGE23_2A_SUMMARY_FILE = "xunce-stage23-2b-rerun-stage23-2a-summary.json"
ROUTING_FILE = "xunce-stage23-5a-next-stage-routing.json"
REPORT_FILE = "xunce-stage23-5a-report.md"
MANIFEST_FILE = "xunce-stage23-5a-manifest.json"

REPAIRED_23_2A_CONFIG = "xunce-stage23-5a-repaired-stage23-2a-config.json"
REPAIRED_23_2B_CONFIG = "xunce-stage23-5a-repaired-stage23-2b-config.json"
REPAIRED_23_2_CONFIG = "xunce-stage23-5a-repaired-stage23-2-config.json"
REPAIRED_23_3_CONFIG = "xunce-stage23-5a-repaired-stage23-3-config.json"
REPAIRED_23_4_CONFIG = "xunce-stage23-5a-repaired-stage23-4-config.json"
REPAIRED_23_5_CONFIG = "xunce-stage23-5a-repaired-stage23-5-config.json"

NESTED_STAGE23_2B_DIR = "b"
NESTED_STAGE23_2_DIR = "r2"
NESTED_STAGE23_3_DIR = "r3"
NESTED_STAGE23_4_DIR = "r4"
NESTED_STAGE23_5_DIR = "r5"

ROUTE_PRESERVE = "preserve_stage23_5_existing_route"
ROUTE_EXPAND_ROI = "expand_or_relocate_stage23_high_res_roi_windows"
ROUTE_REQUIRED_INPUTS = "rerun_stage23_5_required_inputs"
ROUTE_SIGNAL = "repair_stage23_slope_theta_policy_update_signal_strength"
ROUTE_CREDIT = "repair_stage23_slope_theta_credit_assignment"
ROUTE_STAGE23_6 = "run_stage23_6_slope_obstacle_aware_theta_multi_seed_ppo_pilot"
ROUTE_BOUNDARY = "resolve_stage23_5a_boundary_rejections"

BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)

SHORT_MARKERS = (
    "scenario_count_short",
    "xunce_episode_count_short",
    "high_fidelity_roi_expansion_slice_count_short",
)
ALLOWED_REQUIRED_INPUT_REASON_CODES = {
    "stage21_5_offline_evaluation_execution_incomplete",
    "offline_evaluation_execution_incomplete",
}
NON_ROI_INPUT_BLOCKER_COUNT_FIELDS = (
    "slope_theta_inference_required_field_missing_count",
    "coverage_source_mismatch_count",
    "max_traversable_slope_mismatch_count",
    "obstacle_contract_mismatch_count",
    "pre_duplicate_strong_key_count",
    "post_duplicate_strong_key_count",
    "hard_risk_violation_count",
    "mask_violation_count",
    "path_planning_failure_count",
    "open_grid_fallback_count",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage23.5A high-res ROI input repair.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    summary = run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage(
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

    prior_stage23_5_summary = _read_json_if_exists(Path(config["stage23_5_root"]) / stage23_5.SUMMARY_FILE)
    prior_audit = _prior_stage23_5_input_audit(prior_stage23_5_summary)
    roi_audit = _empty_roi_audit(int(config["min_valid_slice_count"]))
    stage23_2b_summary: dict[str, Any] = {}
    stage23_2_summary: dict[str, Any] = {}
    stage23_3_summary: dict[str, Any] = {}
    stage23_4_summary: dict[str, Any] = {}
    stage23_5_summary: dict[str, Any] = {}

    if not boundary_reasons and prior_audit["is_stage23_5_required_input_short_blocker"]:
        roi_audit = _select_valid_roi_windows(config, output_root=output_root, repo_root=repo_root)
        if roi_audit["valid_window_count"] >= int(config["min_valid_slice_count"]):
            stage23_2b_summary, stage23_2_summary, stage23_3_summary, stage23_4_summary, stage23_5_summary = _run_repaired_chain(
                config,
                roi_audit["selected_roi_windows"],
                output_root=output_root,
                repo_root=repo_root,
            )

    status, route, route_reason = _route(
        boundary_reasons=boundary_reasons,
        prior_audit=prior_audit,
        roi_audit=roi_audit,
        stage23_2b_summary=stage23_2b_summary,
        stage23_2_summary=stage23_2_summary,
        stage23_3_summary=stage23_3_summary,
        stage23_4_summary=stage23_4_summary,
        stage23_5_summary=stage23_5_summary,
    )
    blockers = _blocking_reasons(
        boundary_reasons=boundary_reasons,
        prior_audit=prior_audit,
        roi_audit=roi_audit,
        stage23_2b_summary=stage23_2b_summary,
        stage23_2_summary=stage23_2_summary,
        stage23_3_summary=stage23_3_summary,
        stage23_4_summary=stage23_4_summary,
        stage23_5_summary=stage23_5_summary,
        route=route,
    )

    final_stage23_5 = stage23_5_summary or {}
    stage23_2a_rerun_summary = _read_json_if_exists(output_root / NESTED_STAGE23_2B_DIR / RERUN_STAGE23_2A_SUMMARY_FILE)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blockers,
        "stage23_5_root": config["stage23_5_root"],
        "prior_stage23_5_status": prior_stage23_5_summary.get("status"),
        "prior_stage23_5_next_required_change": prior_stage23_5_summary.get("next_required_change"),
        "prior_stage23_5_required_input_short_blocker": prior_audit["is_stage23_5_required_input_short_blocker"],
        "min_valid_slice_count": int(config["min_valid_slice_count"]),
        "candidate_roi_window_count": roi_audit["candidate_window_count"],
        "valid_roi_window_count": roi_audit["valid_window_count"],
        "selected_roi_window_count": len(roi_audit.get("selected_roi_windows", [])),
        "repaired_high_res_roi_expansion_slice_count": int(stage23_2a_rerun_summary.get("high_res_slice_count", 0) or 0),
        "repaired_high_res_roi_expansion_scenario_count": int(
            stage23_2a_rerun_summary.get("scenario_count", stage23_2a_rerun_summary.get("high_res_scenario_count", 0)) or 0
        ),
        "repaired_stage23_2b_status": stage23_2b_summary.get("status"),
        "repaired_stage23_2_status": stage23_2_summary.get("status"),
        "repaired_stage23_3_status": stage23_3_summary.get("status"),
        "repaired_stage23_4_status": stage23_4_summary.get("status"),
        "repaired_stage23_5_status": final_stage23_5.get("status"),
        "repaired_stage23_5_next_required_change": final_stage23_5.get("next_required_change"),
        "stage23_5_execution_incomplete": bool(final_stage23_5.get("stage21_5_execution_incomplete")),
        "stage23_5_execution_reason_codes": list(final_stage23_5.get("stage21_5_execution_reason_codes") or []),
        "stage23_5_action_probability_audit_is_partial_diagnostic": bool(
            final_stage23_5.get("action_probability_audit_is_partial_diagnostic", True)
        ),
        "strong_state_join_available_count": int(final_stage23_5.get("strong_state_join_available_count", 0) or 0),
        "selected_viewpoint_changed_count": int(final_stage23_5.get("selected_viewpoint_changed_count", 0) or 0),
        "selected_theta_changed_count": int(final_stage23_5.get("selected_theta_changed_count", 0) or 0),
        "selected_action_changed_count": int(final_stage23_5.get("selected_action_changed_count", 0) or 0),
        "mean_abs_probability_delta": _finite_or_default(final_stage23_5.get("mean_abs_probability_delta"), 0.0),
        "final_coverage_delta": _finite_or_default(final_stage23_5.get("final_coverage_delta"), 0.0),
        "coverage_auc_delta": _finite_or_default(final_stage23_5.get("coverage_auc_delta"), 0.0),
        "path_cost_delta": _finite_or_default(final_stage23_5.get("path_cost_delta"), 0.0),
        "hard_risk_violation_count": int(final_stage23_5.get("hard_risk_violation_count", 0) or 0),
        "mask_violation_count": int(final_stage23_5.get("mask_violation_count", 0) or 0),
        "path_planning_failure_count": int(final_stage23_5.get("path_planning_failure_count", 0) or 0),
        "open_grid_fallback_count": int(final_stage23_5.get("open_grid_fallback_count", 0) or 0),
        "platform_contract_hash": _first_present(
            final_stage23_5.get("platform_contract_hash"),
            stage23_3_summary.get("platform_contract_hash"),
            stage23_2_summary.get("platform_contract_hash"),
            stage23_2b_summary.get("platform_contract_hash"),
        ),
        "max_traversable_slope_deg": 30.0,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "stage23_5a_authorized": False,
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
        "stage23_5a_authorized": False,
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
        "roi_window_selection_audit": str(output_root / ROI_AUDIT_FILE),
        "rerun_stage23_2b_summary": str(output_root / RERUN_23_2B_FILE),
        "rerun_stage23_2_summary": str(output_root / RERUN_23_2_FILE),
        "rerun_stage23_3_summary": str(output_root / RERUN_23_3_FILE),
        "rerun_stage23_4_summary": str(output_root / RERUN_23_4_FILE),
        "rerun_stage23_5_summary": str(output_root / RERUN_23_5_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / ROI_AUDIT_FILE, roi_audit)
    _write_json(output_root / RERUN_23_2B_FILE, stage23_2b_summary)
    _write_json(output_root / RERUN_23_2_FILE, stage23_2_summary)
    _write_json(output_root / RERUN_23_3_FILE, stage23_3_summary)
    _write_json(output_root / RERUN_23_4_FILE, stage23_4_summary)
    _write_json(output_root / RERUN_23_5_FILE, stage23_5_summary)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _run_repaired_chain(
    config: dict[str, Any],
    selected_windows: list[dict[str, Any]],
    *,
    output_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage23_2a_config_path = output_root / REPAIRED_23_2A_CONFIG
    stage23_2b_config_path = output_root / REPAIRED_23_2B_CONFIG
    stage23_2_config_path = output_root / REPAIRED_23_2_CONFIG
    stage23_3_config_path = output_root / REPAIRED_23_3_CONFIG
    stage23_4_config_path = output_root / REPAIRED_23_4_CONFIG
    stage23_5_config_path = output_root / REPAIRED_23_5_CONFIG

    stage23_2a_config = _load_template(config["stage23_2a_base_config"], repo_root)
    stage23_2a_config.update(
        {
            "primary_manifest": config["primary_manifest"],
            "raw_data_root": config["raw_data_root"],
            "download_missing": bool(config.get("download_missing", True)),
            "high_res_output_subdir": "hr",
            "path_planner_sidecar_subdir": "ps",
            "stage23_1_output_subdir": "s1",
            "stage23_0a_output_subdir": "a",
            "stage23_0a_high_fidelity_output_subdir": "h",
            "roi_windows": selected_windows,
            "run_stage23_1_smoke": True,
            "max_traversable_slope_deg": 30.0,
            "stage23_2a_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    _write_json(stage23_2a_config_path, stage23_2a_config)

    stage23_2b_config = _load_template(config["stage23_2b_base_config"], repo_root)
    stage23_2b_config.update(
        {
            "stage23_2a_config": str(stage23_2a_config_path),
            "stage23_2a_default_output_subdir": "a30",
            "stage23_2a_sensitivity_output_subdir": "a20",
            "run_stage23_2a_smoke": True,
            "run_sensitivity_smoke": bool(config.get("run_sensitivity_smoke", True)),
            "run_stage23_1_smoke": True,
            "default_threshold_deg": 30.0,
            "sensitivity_threshold_deg": 20.0,
            "stage23_2b_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    _write_json(stage23_2b_config_path, stage23_2b_config)
    stage23_2b_summary = stage23_2b.run_xunce_stage23_2b_platform_geometry_sensor_contract_alignment(
        config_path=stage23_2b_config_path,
        output_root=output_root / NESTED_STAGE23_2B_DIR,
        repo_root=repo_root,
    )
    if stage23_2b_summary.get("status") != "passed":
        return stage23_2b_summary, {}, {}, {}, {}

    stage23_2_config = _load_template(config["stage23_2_base_config"], repo_root)
    stage23_2_config.update(
        {
            "stage23_2b_root": str(output_root / NESTED_STAGE23_2B_DIR),
            "max_traversable_slope_deg": 30.0,
            "stage23_2_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    _write_json(stage23_2_config_path, stage23_2_config)
    stage23_2_summary = stage23_2.run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract(
        config_path=stage23_2_config_path,
        output_root=output_root / NESTED_STAGE23_2_DIR,
        repo_root=repo_root,
    )
    if stage23_2_summary.get("status") != "passed":
        return stage23_2b_summary, stage23_2_summary, {}, {}, {}

    stage23_3_config = _load_template(config["stage23_3_base_config"], repo_root)
    stage23_3_config.update({"stage23_2_root": str(output_root / NESTED_STAGE23_2_DIR), "stage23_3_authorized": False})
    _write_json(stage23_3_config_path, stage23_3_config)
    stage23_3_summary = stage23_3.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke(
        config_path=stage23_3_config_path,
        output_root=output_root / NESTED_STAGE23_3_DIR,
        repo_root=repo_root,
    )
    if stage23_3_summary.get("status") != "passed":
        return stage23_2b_summary, stage23_2_summary, stage23_3_summary, {}, {}

    stage23_4_config = _load_template(config["stage23_4_base_config"], repo_root)
    stage23_4_config.update({"stage23_3_root": str(output_root / NESTED_STAGE23_3_DIR), "stage23_4_authorized": False})
    _write_json(stage23_4_config_path, stage23_4_config)
    stage23_4_summary = stage23_4.run_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke(
        config_path=stage23_4_config_path,
        output_root=output_root / NESTED_STAGE23_4_DIR,
        repo_root=repo_root,
    )
    if stage23_4_summary.get("status") != "passed":
        return stage23_2b_summary, stage23_2_summary, stage23_3_summary, stage23_4_summary, {}

    stage23_5_config = _load_template(config["stage23_5_base_config"], repo_root)
    stage23_5_config.update({"stage23_4_root": str(output_root / NESTED_STAGE23_4_DIR), "stage23_5_authorized": False})
    _write_json(stage23_5_config_path, stage23_5_config)
    stage23_5_summary = stage23_5.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke(
        config_path=stage23_5_config_path,
        output_root=output_root / NESTED_STAGE23_5_DIR,
        repo_root=repo_root,
    )
    return stage23_2b_summary, stage23_2_summary, stage23_3_summary, stage23_4_summary, stage23_5_summary


def _select_valid_roi_windows(config: dict[str, Any], *, output_root: Path, repo_root: Path) -> dict[str, Any]:
    candidates = list(config["roi_candidate_windows"])
    selected: list[dict[str, Any]] = []
    attempted: list[dict[str, Any]] = []
    product_error: str | None = None

    try:
        stage23_2a_config = _load_template(config["stage23_2a_base_config"], repo_root)
        manifest_path = _resolve_path(Path(str(stage23_2a_config.get("primary_manifest", config["primary_manifest"]))), repo_root)
        manifest = _read_json(manifest_path)
        products, _, _ = stage23_2a._prepare_products(
            manifest,
            output_root=output_root / "roi_window_precheck_products",
            raw_data_root=_resolve_path(Path(str(stage23_2a_config.get("raw_data_root", config["raw_data_root"]))), repo_root),
            download_missing=bool(stage23_2a_config.get("download_missing", config.get("download_missing", True))),
        )
        dem = products["dem"]
    except (OSError, ValueError, KeyError, GeoTiffDecodeUnavailable) as exc:
        product_error = str(exc)
        dem = None

    for index, window in enumerate(candidates):
        row = {
            "index": index,
            "roi_window": dict(window),
            "selected": False,
            "valid": False,
            "reason": None,
        }
        if dem is None:
            row["reason"] = "geotiff_product_prepare_failed"
            attempted.append(row)
            continue
        try:
            geotiff_window = read_geotiff_window(
                dem.path,
                x=int(window["x"]),
                y=int(window["y"]),
                width=int(window["width"]),
                height=int(window["height"]),
                resolution_m=dem.resolution_m,
                projection=dem.projection,
                nodata_value=window.get("dem_nodata_value", stage23_2a_config.get("dem_nodata_value")),
            )
        except (OSError, ValueError, GeoTiffDecodeUnavailable) as exc:
            row["reason"] = f"window_decode_failed:{exc}"
            attempted.append(row)
            continue
        total_cells = geotiff_window.width * geotiff_window.height
        nodata_count = sum(1 for mask_row in geotiff_window.nodata_mask for value in mask_row if value)
        row.update(
            {
                "valid": nodata_count < total_cells,
                "width": geotiff_window.width,
                "height": geotiff_window.height,
                "nodata_count": nodata_count,
                "total_cell_count": total_cells,
                "reader_backend": geotiff_window.reader_backend,
            }
        )
        if nodata_count >= total_cells:
            row["reason"] = "window_all_nodata"
        else:
            row["reason"] = "valid"
            repaired = dict(window)
            repaired.setdefault("roi_name", f"stage23_5a_roi_{len(selected):03d}")
            repaired.setdefault("split", "train")
            repaired.setdefault("candidate_count", int(config["candidate_count"]))
            repaired.setdefault("start_cell", [0, 0])
            selected.append(repaired)
            row["selected"] = len(selected) <= int(config["min_valid_slice_count"])
        attempted.append(row)

    selected = selected[: int(config["min_valid_slice_count"])]
    return {
        "schema_version": ROI_AUDIT_SCHEMA_VERSION,
        "candidate_window_count": len(candidates),
        "valid_window_count": sum(1 for row in attempted if row.get("valid") is True),
        "selected_window_count": len(selected),
        "min_valid_slice_count": int(config["min_valid_slice_count"]),
        "product_prepare_error": product_error,
        "attempted_windows": attempted,
        "selected_roi_windows": selected,
    }


def _prior_stage23_5_input_audit(summary: dict[str, Any]) -> dict[str, Any]:
    route = summary.get("next_required_change")
    reason_codes = [str(value) for value in summary.get("stage21_5_execution_reason_codes") or summary.get("blocking_reason_codes") or []]
    short_codes = [code for code in reason_codes if any(marker in code for marker in SHORT_MARKERS)]
    non_short_reason_codes = [
        code
        for code in reason_codes
        if code not in ALLOWED_REQUIRED_INPUT_REASON_CODES and not any(marker in code for marker in SHORT_MARKERS)
    ]
    non_roi_blocker_counts = {
        field: int(summary.get(field, 0) or 0)
        for field in NON_ROI_INPUT_BLOCKER_COUNT_FIELDS
    }
    has_non_roi_blockers = bool(non_short_reason_codes) or any(value > 0 for value in non_roi_blocker_counts.values())
    return {
        "prior_summary_present": bool(summary),
        "prior_status": summary.get("status"),
        "prior_route": route,
        "reason_codes": reason_codes,
        "short_reason_codes": short_codes,
        "non_short_reason_codes": non_short_reason_codes,
        "non_roi_blocker_counts": non_roi_blocker_counts,
        "is_stage23_5_required_input_short_blocker": bool(
            summary
            and summary.get("status") == "failed"
            and route == ROUTE_REQUIRED_INPUTS
            and summary.get("stage21_5_execution_incomplete") is True
            and short_codes
            and not has_non_roi_blockers
        ),
    }


def _route(
    *,
    boundary_reasons: list[str],
    prior_audit: dict[str, Any],
    roi_audit: dict[str, Any],
    stage23_2b_summary: dict[str, Any],
    stage23_2_summary: dict[str, Any],
    stage23_3_summary: dict[str, Any],
    stage23_4_summary: dict[str, Any],
    stage23_5_summary: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "Stage23.5A boundary fields are not clean"
    if not prior_audit["is_stage23_5_required_input_short_blocker"]:
        return "failed", ROUTE_PRESERVE, "Stage23.5 prior blocker is not the scenario/episode-short required-input route"
    if roi_audit["valid_window_count"] < roi_audit["min_valid_slice_count"]:
        return "failed", ROUTE_EXPAND_ROI, "fewer than two valid high-res ROI windows are available"
    if stage23_2b_summary.get("status") != "passed":
        return "failed", str(stage23_2b_summary.get("next_required_change") or "repair_stage23_2b_high_res_roi_rerun"), "Stage23.2B rerun did not pass"
    if stage23_2_summary.get("status") != "passed":
        return "failed", str(stage23_2_summary.get("next_required_change") or "repair_stage23_2_slope_reward_contract"), "Stage23.2 rerun did not pass"
    if stage23_3_summary.get("status") != "passed":
        return "failed", str(stage23_3_summary.get("next_required_change") or "repair_stage23_3_collector_contract"), "Stage23.3 rerun did not pass"
    if stage23_4_summary.get("status") != "passed":
        return "failed", str(stage23_4_summary.get("next_required_change") or "repair_stage23_4_update_smoke"), "Stage23.4 rerun did not pass"
    if stage23_5_summary.get("next_required_change") == ROUTE_REQUIRED_INPUTS and _stage23_5_has_short_blocker(stage23_5_summary):
        return "failed", ROUTE_REQUIRED_INPUTS, "Stage23.5 still has scenario/episode-short inputs after ROI repair"
    if stage23_5_summary.get("status") == "passed" and stage23_5_summary.get("next_required_change") == ROUTE_STAGE23_6:
        return "passed", ROUTE_STAGE23_6, "Stage23.5 rerun produced complete pre/post trajectory smoke inputs"
    route = str(stage23_5_summary.get("next_required_change") or ROUTE_SIGNAL)
    return "failed", route, "Stage23.5 rerun completed and preserves its diagnostic route"


def _blocking_reasons(
    *,
    boundary_reasons: list[str],
    prior_audit: dict[str, Any],
    roi_audit: dict[str, Any],
    stage23_2b_summary: dict[str, Any],
    stage23_2_summary: dict[str, Any],
    stage23_3_summary: dict[str, Any],
    stage23_4_summary: dict[str, Any],
    stage23_5_summary: dict[str, Any],
    route: str,
) -> list[str]:
    reasons: list[str] = list(boundary_reasons)
    if route == ROUTE_PRESERVE:
        reasons.append("stage23_5_prior_blocker_not_scenario_episode_short")
        reasons.extend(f"prior_stage23_5_{code}" for code in prior_audit.get("non_short_reason_codes", []))
        for field, value in (prior_audit.get("non_roi_blocker_counts") or {}).items():
            if int(value or 0) > 0:
                reasons.append(f"prior_stage23_5_{field}_nonzero")
    if route == ROUTE_EXPAND_ROI:
        reasons.append("valid_high_res_roi_window_count_below_minimum")
        if roi_audit.get("product_prepare_error"):
            reasons.append("geotiff_product_prepare_failed")
    for name, summary in (
        ("stage23_2b", stage23_2b_summary),
        ("stage23_2", stage23_2_summary),
        ("stage23_3", stage23_3_summary),
        ("stage23_4", stage23_4_summary),
    ):
        if summary and summary.get("status") != "passed":
            reasons.extend([f"{name}_{code}" for code in summary.get("blocking_reason_codes", [])])
    if stage23_5_summary and stage23_5_summary.get("status") != "passed":
        reasons.extend([f"stage23_5_{code}" for code in stage23_5_summary.get("blocking_reason_codes", [])])
    if not prior_audit.get("prior_summary_present"):
        reasons.append("missing_prior_stage23_5_summary")
    return _unique(reasons)


def _stage23_5_has_short_blocker(summary: dict[str, Any]) -> bool:
    codes = [str(value) for value in summary.get("stage21_5_execution_reason_codes") or summary.get("blocking_reason_codes") or []]
    return any(any(marker in code for marker in SHORT_MARKERS) for code in codes)


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    payload.setdefault("stage23_5_root", "D:/CodexDownloads/lunar-path-planning/stage23_slope_obstacle_aware_theta_reward/outputs/path_feedback_batch_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke_v1")
    payload.setdefault("stage23_2a_base_config", "configs/xunce_stage23_2a_high_resolution_terrain_data_ingestion_v1.json")
    payload.setdefault("stage23_2b_base_config", "configs/xunce_stage23_2b_platform_geometry_sensor_contract_alignment_v1.json")
    payload.setdefault("stage23_2_base_config", "configs/xunce_stage23_2_slope_obstacle_aware_theta_reward_contract_v1.json")
    payload.setdefault("stage23_3_base_config", "configs/xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke_v1.json")
    payload.setdefault("stage23_4_base_config", "configs/xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke_v1.json")
    payload.setdefault("stage23_5_base_config", "configs/xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke_v1.json")
    payload.setdefault("primary_manifest", "model-explorer/data/manifests/lunar_south_pole_usgs_lro_dem_slope_4m.json")
    payload.setdefault("raw_data_root", "D:/CodexDownloads/lunar-path-planning/data/raw/high_resolution_lunar_terrain")
    payload.setdefault("download_missing", True)
    payload.setdefault("candidate_count", 8)
    payload.setdefault("min_valid_slice_count", 2)
    payload.setdefault("run_sensitivity_smoke", True)
    payload.setdefault("roi_candidate_windows", _default_roi_candidate_windows())
    return payload


def _default_roi_candidate_windows() -> list[dict[str, Any]]:
    return [
        {"roi_name": "stage23_5a_roi_8000_8000", "split": "train", "x": 8000, "y": 8000, "width": 32, "height": 32, "candidate_count": 8, "seed": 230201, "start_cell": [0, 0]},
        {"roi_name": "stage23_5a_roi_8032_8000", "split": "train", "x": 8032, "y": 8000, "width": 32, "height": 32, "candidate_count": 8, "seed": 230202, "start_cell": [0, 0]},
        {"roi_name": "stage23_5a_roi_8000_8032", "split": "train", "x": 8000, "y": 8032, "width": 32, "height": 32, "candidate_count": 8, "seed": 230203, "start_cell": [0, 0]},
        {"roi_name": "stage23_5a_roi_8032_8032", "split": "train", "x": 8032, "y": 8032, "width": 32, "height": 32, "candidate_count": 8, "seed": 230204, "start_cell": [0, 0]},
        {"roi_name": "stage23_5a_roi_8064_8000", "split": "train", "x": 8064, "y": 8000, "width": 32, "height": 32, "candidate_count": 8, "seed": 230205, "start_cell": [0, 0]},
    ]


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction_nonzero")
    return reasons


def _empty_roi_audit(min_valid_slice_count: int = 2) -> dict[str, Any]:
    return {
        "schema_version": ROI_AUDIT_SCHEMA_VERSION,
        "candidate_window_count": 0,
        "valid_window_count": 0,
        "selected_window_count": 0,
        "min_valid_slice_count": int(min_valid_slice_count),
        "product_prepare_error": None,
        "attempted_windows": [],
        "selected_roi_windows": [],
    }


def _load_template(path: str, repo_root: Path) -> dict[str, Any]:
    return _read_json(_resolve_path(Path(path), repo_root))


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _nested_int(payload: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        if payload.get(key) is not None:
            return int(payload.get(key) or 0)
    return 0


def _finite_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage23.5A Repair Required Inputs / High-Res ROI Coverage",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- prior_stage23_5_required_input_short_blocker: `{summary['prior_stage23_5_required_input_short_blocker']}`",
            f"- selected_roi_window_count: `{summary['selected_roi_window_count']}`",
            f"- repaired_stage23_5_status: `{summary['repaired_stage23_5_status']}`",
            f"- stage23_5_action_probability_audit_is_partial_diagnostic: `{summary['stage23_5_action_probability_audit_is_partial_diagnostic']}`",
            f"- strong_state_join_available_count: `{summary['strong_state_join_available_count']}`",
            f"- selected_viewpoint_changed_count: `{summary['selected_viewpoint_changed_count']}`",
            f"- final_coverage_delta: `{summary['final_coverage_delta']}`",
            f"- coverage_auc_delta: `{summary['coverage_auc_delta']}`",
            "",
            "Stage23.5A only repairs bounded high-res ROI input coverage and reruns the existing Stage23 chain. It does not publish checkpoints, replace the default policy, connect an executor, start canary traffic, or change PPO/network/default A*/candidate-generation semantics.",
            "",
        ]
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
