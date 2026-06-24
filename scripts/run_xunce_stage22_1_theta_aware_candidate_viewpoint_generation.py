from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by script execution
    from xunce_theta_viewpoint_candidates import (
        expand_theta_aware_candidates,
        row_has_theta_viewpoint_contract,
    )
    import run_xunce_stage22_0_theta_aware_sensor_action_space_contract as stage22_0
except ModuleNotFoundError:  # pragma: no cover
    from scripts.xunce_theta_viewpoint_candidates import (
        expand_theta_aware_candidates,
        row_has_theta_viewpoint_contract,
    )
    import scripts.run_xunce_stage22_0_theta_aware_sensor_action_space_contract as stage22_0


CONFIG_SCHEMA_VERSION = "xunce-stage22-1-theta-aware-candidate-viewpoint-generation-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage22-1-summary/v1"
VIEWPOINT_ROW_SCHEMA_VERSION = "xunce-stage22-1-viewpoint-candidate-audit-row/v1"
BATCH_CONTRACT_SCHEMA_VERSION = "xunce-stage22-1-batch-contract-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage22-1-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage22-1-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage22_1_theta_aware_candidate_viewpoint_generation_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage22_theta_aware_sensor_action_space/"
    "outputs/path_feedback_batch_xunce_stage22_1_theta_aware_candidate_viewpoint_generation_v1"
)

SUMMARY_FILE = "xunce-stage22-1-summary.json"
VIEWPOINT_AUDIT_FILE = "xunce-stage22-1-viewpoint-candidate-audit.jsonl"
BATCH_CONTRACT_FILE = "xunce-stage22-1-batch-contract-audit.json"
ROUTING_FILE = "xunce-stage22-1-next-stage-routing.json"
REPORT_FILE = "xunce-stage22-1-report.md"
MANIFEST_FILE = "xunce-stage22-1-manifest.json"

ROUTE_INPUTS = "rerun_stage22_1_required_inputs"
ROUTE_VIEWPOINT_CONTRACT = "repair_stage22_1_viewpoint_contract"
ROUTE_BATCH_CONTRACT = "repair_stage22_1_stage21_batch_viewpoint_contract"
ROUTE_STAGE22_2 = "run_stage22_2_theta_aware_coverage_reward_contract"
ROUTE_BOUNDARY = "resolve_stage22_1_boundary_rejections"

BOUNDARY_FIELDS = (
    "stage22_1_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage22.1 theta-aware candidate viewpoint generation audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage22_1_theta_aware_candidate_viewpoint_generation(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    return 1 if summary["status"] == "failed" else 0


def run_xunce_stage22_1_theta_aware_candidate_viewpoint_generation(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path, repo_root)
    boundary_reasons = _boundary_rejections(config)
    input_reasons = _input_rejections(config)
    roots, root_reasons = stage22_0._discover_stage21_roots(config)
    rows, row_reasons = stage22_0._load_stage21_rows(roots)
    input_reasons.extend(root_reasons)
    input_reasons.extend(row_reasons)
    if not rows:
        input_reasons.append("stage21_trainable_rows_missing")

    viewpoint_rows, batch_contract = _audit_viewpoint_contract(rows, config)
    status, route, route_reason = _route(input_reasons, boundary_reasons, batch_contract)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": "xunce-stage22-1-theta-aware-candidate-viewpoint-generation",
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "input_reason_codes": sorted(set(input_reasons)),
        "boundary_reason_codes": sorted(set(boundary_reasons)),
        "stage22_0_root": config["stage22_0_root"],
        "audited_transition_count": len(rows),
        "viewpoint_candidate_audit_row_count": len(viewpoint_rows),
        "theta_bin_count": int(config["theta_bin_count"]),
        "theta_step_deg": int(config["theta_step_deg"]),
        "sensor_model_id": config["sensor_model_id"],
        "sensor_fov_deg": float(config["sensor_fov_deg"]),
        "sensor_range_cells": int(config["sensor_range_cells"]),
        **{k: batch_contract[k] for k in (
            "point_only_input_row_count",
            "theta_contract_input_row_count",
            "expanded_viewpoint_row_count",
            "hash_missing_theta_count",
            "mask_length_mismatch_count",
            "stage21_batch_theta_incompatible_after_expansion",
        )},
        "release_or_training_authorized": False,
        "stage22_1_authorized": False,
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
        "stage22_1_authorized": False,
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
        "viewpoint_candidate_audit": str(output_root / VIEWPOINT_AUDIT_FILE),
        "batch_contract_audit": str(output_root / BATCH_CONTRACT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_jsonl(output_root / VIEWPOINT_AUDIT_FILE, viewpoint_rows)
    _write_json(output_root / BATCH_CONTRACT_FILE, batch_contract)
    _write_json(output_root / ROUTING_FILE, routing)
    (output_root / REPORT_FILE).write_text(_render_report(summary, batch_contract), encoding="utf-8")
    _write_json(output_root / MANIFEST_FILE, manifest)
    return summary


def _audit_viewpoint_contract(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    point_only_rows = 0
    theta_contract_rows = 0
    mask_mismatch_count = 0
    hash_missing_theta_count = 0
    expanded_viewpoint_count = 0
    theta_feature_missing_count = 0
    max_rows = int(config["max_viewpoint_candidate_audit_rows"])
    for row in rows:
        has_theta = row_has_theta_viewpoint_contract(row)
        theta_contract_rows += int(has_theta)
        point_only_rows += int(not has_theta)
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        cells = stage22_0._candidate_cells(row)
        action_mask = _bool_list(info.get("action_mask"), len(cells))
        candidates = [
            {
                "cell": list(cell),
                "candidate_cell": list(cell),
                "action_index": index,
                "path_cost": stage22_0._candidate_path_cost(row, index),
                "reachable": True,
                "risk": None,
            }
            for index, cell in enumerate(cells)
        ]
        expansion = expand_theta_aware_candidates(
            candidates,
            current_cell=stage22_0._cell_tuple(info.get("current_cell_before")),
            covered_cells=set(),
            config=config,
            base_candidate_set_hash=info.get("candidate_set_hash"),
        )
        expanded = expansion["candidates"]
        expanded_viewpoint_count += len(expanded)
        expanded_mask = [
            bool(action_mask[item["base_candidate_index"]])
            if int(item["base_candidate_index"]) < len(action_mask)
            else False
            for item in expanded
        ]
        if len(expanded_mask) != len(expanded) or len(action_mask) != len(cells):
            mask_mismatch_count += 1
        if not expanded or not expansion["candidate_set_hash"]:
            hash_missing_theta_count += 1
        if any(candidate.get("theta_feature_available") is not True for candidate in expanded):
            theta_feature_missing_count += 1
        for candidate in expanded[: max(0, max_rows - len(audit_rows))]:
            audit_rows.append(
                {
                    "schema_version": VIEWPOINT_ROW_SCHEMA_VERSION,
                    "scenario_id": row.get("scenario_id") or info.get("scenario_id"),
                    "step_index": row.get("step_index") or info.get("step_index"),
                    "candidate_cell": candidate.get("candidate_cell"),
                    "candidate_theta_deg": candidate.get("candidate_theta_deg"),
                    "candidate_viewpoint": candidate.get("candidate_viewpoint"),
                    "base_candidate_index": candidate.get("base_candidate_index"),
                    "viewpoint_index": candidate.get("viewpoint_index"),
                    "base_candidate_set_hash": candidate.get("base_candidate_set_hash"),
                    "candidate_set_hash": expansion["candidate_set_hash"],
                    "sensor_model_id": candidate.get("sensor_model_id"),
                    "sensor_fov_deg": candidate.get("sensor_fov_deg"),
                    "sensor_range_cells": candidate.get("sensor_range_cells"),
                    "theta_visible_cell_count": candidate.get("theta_visible_cell_count"),
                    "theta_new_visible_cell_count": candidate.get("theta_new_visible_cell_count"),
                    "theta_coverage_hash": candidate.get("theta_coverage_hash"),
                    "theta_coverage_gain_per_path_cost": candidate.get("theta_coverage_gain_per_path_cost"),
                    "action_mask_valid": bool(expanded_mask[candidate["viewpoint_index"]]),
                    "theta_feature_available": candidate.get("theta_feature_available"),
                    "sin_theta": candidate.get("sin_theta"),
                    "cos_theta": candidate.get("cos_theta"),
                    "theta_deg_norm": candidate.get("theta_deg_norm"),
                }
            )
    expected_multiplier = int(config["theta_bin_count"])
    batch_contract = {
        "schema_version": BATCH_CONTRACT_SCHEMA_VERSION,
        "audited_transition_count": len(rows),
        "point_only_input_row_count": point_only_rows,
        "theta_contract_input_row_count": theta_contract_rows,
        "expected_theta_multiplier": expected_multiplier,
        "expanded_viewpoint_row_count": expanded_viewpoint_count,
        "mask_length_mismatch_count": mask_mismatch_count,
        "hash_missing_theta_count": hash_missing_theta_count,
        "theta_feature_missing_count": theta_feature_missing_count,
        "stage21_batch_theta_incompatible_before_expansion": point_only_rows > 0,
        "stage21_batch_theta_incompatible_after_expansion": bool(
            mask_mismatch_count or hash_missing_theta_count or theta_feature_missing_count
        ),
        "action_mask_semantics": "viewpoint-level mask expanded from base candidate mask",
        "sampling_mask_semantics": "viewpoint-level sampling mask must match expanded viewpoint candidates",
        "hard_risk_clean_mask_semantics": "viewpoint-level hard-risk mask inherits the base (x,y) path-risk result",
        "candidate_set_hash_includes_theta": hash_missing_theta_count == 0 and expanded_viewpoint_count > 0,
        "ppo_network_architecture_changed": False,
        "default_astar_changed": False,
    }
    return audit_rows, batch_contract


def _route(input_reasons: list[str], boundary_reasons: list[str], contract: dict[str, Any]) -> tuple[str, str, str]:
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage22_1_inputs_missing_or_untrusted"
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage22_1_boundary_rejected"
    if contract["hash_missing_theta_count"] or contract["mask_length_mismatch_count"] or contract["theta_feature_missing_count"]:
        return "failed", ROUTE_VIEWPOINT_CONTRACT, "viewpoint_hash_mask_or_theta_feature_contract_failed"
    if contract["stage21_batch_theta_incompatible_after_expansion"]:
        return "failed", ROUTE_BATCH_CONTRACT, "stage21_batch_viewpoint_contract_still_incompatible"
    return "passed", ROUTE_STAGE22_2, "theta_aware_viewpoint_candidate_and_batch_contract_available"


def _input_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    root = Path(config["stage22_0_root"])
    summary_path = root / "xunce-stage22-0-summary.json"
    if not summary_path.is_file():
        reasons.append("missing_stage22_0_summary")
        return reasons
    summary = _read_json(summary_path)
    if summary.get("status") != "passed":
        reasons.append("stage22_0_not_passed")
    if summary.get("next_required_change") != "implement_stage22_1_theta_aware_candidate_viewpoint_generation":
        reasons.append("stage22_0_route_not_stage22_1")
    if summary.get("theta_material_to_coverage") is not True:
        reasons.append("stage22_0_theta_not_material")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is not False]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _load_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    path = config_path if config_path.is_absolute() else repo_root / config_path
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported config schema_version: {payload.get('schema_version')}")
    defaults = {
        "stage22_0_root": (
            "D:/CodexDownloads/lunar-path-planning/stage22_theta_aware_sensor_action_space/"
            "outputs/path_feedback_batch_xunce_stage22_0_theta_aware_sensor_action_space_contract_v1"
        ),
        "stage21_18_root": "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/outputs/path_feedback_batch_xunce_stage21_18_iterative_ppo_learning_curve_audit_v1",
        "stage21_19_root": "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/outputs/path_feedback_batch_xunce_stage21_19_policy_update_signal_source_repair_v1",
        "theta_aware_candidate_viewpoints_enabled": True,
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_model_id": "theta-fov-90-range-radius/v1",
        "sensor_fov_deg": 90.0,
        "sensor_range_cells": 2,
        "coverage_denominator_cells": 1.0,
        "max_viewpoint_candidate_audit_rows": 5000,
        "stage22_1_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    for key in ("stage22_0_root", "stage21_18_root"):
        config[key] = str(_resolve_path(Path(str(config[key])), repo_root))
    if str(config.get("stage21_19_root") or ""):
        config["stage21_19_root"] = str(_resolve_path(Path(str(config["stage21_19_root"])), repo_root))
    else:
        config["stage21_19_root"] = ""
    config["theta_bin_count"] = _positive_int(config["theta_bin_count"], "theta_bin_count")
    config["theta_step_deg"] = _positive_int(config["theta_step_deg"], "theta_step_deg")
    config["sensor_range_cells"] = _positive_int(config["sensor_range_cells"], "sensor_range_cells")
    config["sensor_fov_deg"] = _positive_float(config["sensor_fov_deg"], "sensor_fov_deg")
    config["coverage_denominator_cells"] = _positive_float(config["coverage_denominator_cells"], "coverage_denominator_cells")
    config["max_viewpoint_candidate_audit_rows"] = _positive_int(
        config["max_viewpoint_candidate_audit_rows"],
        "max_viewpoint_candidate_audit_rows",
    )
    return config


def _render_report(summary: dict[str, Any], contract: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage22.1 Theta-Aware Candidate Viewpoint Generation",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- audited_transition_count: `{summary['audited_transition_count']}`",
            f"- viewpoint_candidate_audit_row_count: `{summary['viewpoint_candidate_audit_row_count']}`",
            f"- point_only_input_row_count: `{contract['point_only_input_row_count']}`",
            f"- expanded_viewpoint_row_count: `{contract['expanded_viewpoint_row_count']}`",
            f"- mask_length_mismatch_count: `{contract['mask_length_mismatch_count']}`",
            f"- hash_missing_theta_count: `{contract['hash_missing_theta_count']}`",
            "",
            "Stage22.1 only establishes the `(x,y,theta)` viewpoint candidate contract; paths still plan to `(x,y)`, and PPO/checkpoint publication remain disabled.",
        ]
    ) + "\n"


def _bool_list(value: Any, length: int) -> list[bool]:
    if isinstance(value, list):
        return [bool(item) for item in value]
    return [True] * length


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


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


if __name__ == "__main__":
    raise SystemExit(main())
