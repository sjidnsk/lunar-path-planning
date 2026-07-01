from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3


STAGE_ID = "xunce-stage26-7h-repair-credit-post-update-eval-binding"
CONFIG_SCHEMA_VERSION = "xunce-stage26-7h-repair-credit-post-update-eval-binding-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-7h-summary/v1"
AUDIT_SCHEMA_VERSION = "xunce-stage26-7h-inference-binding-audit/v1"
UNREACHABLE_AUDIT_SCHEMA_VERSION = "xunce-stage26-7h-unreachable-selected-audit/v1"
EFFICIENCY_AUDIT_SCHEMA_VERSION = "xunce-stage26-7h-coverage-efficiency-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-7h-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-7h-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_7h_repair_credit_post_update_eval_binding_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_7h_repair_credit_post_update_eval_binding_v1"
)
DEFAULT_STAGE26_7G_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_7g_repair_behavior_policy_kl_baseline_v1"
)

SUMMARY_FILE = "xunce-stage26-7h-summary.json"
STAGE26_3_CONFIG_FILE = "xunce-stage26-7h-stage26-3-config.json"
STAGE26_3_SUMMARY_FILE = "xunce-stage26-7h-rerun-stage26-3-summary.json"
INFERENCE_AUDIT_FILE = "xunce-stage26-7h-inference-binding-audit.json"
UNREACHABLE_AUDIT_FILE = "xunce-stage26-7h-unreachable-selected-audit.json"
EFFICIENCY_AUDIT_FILE = "xunce-stage26-7h-coverage-efficiency-audit.json"
ROUTING_FILE = "xunce-stage26-7h-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-7h-report.md"
MANIFEST_FILE = "xunce-stage26-7h-manifest.json"

ROUTE_INPUTS = "rerun_stage26_7h_required_inputs"
ROUTE_BINDING = "continue_stage26_7h_eval_binding_repair"
ROUTE_PRE_UNREACHABLE = "repair_stage26_7h_pre_policy_unreachable_baseline"
ROUTE_POST_UNREACHABLE = "repair_stage26_7h_post_policy_unreachable_regression"
ROUTE_STAGE21_5 = "repair_stage26_7h_stage21_5_execution_contract"
ROUTE_MARGIN = "calibrate_stage26_synthetic_discrete_margin_crossing_after_credit"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_MULTI_SEED = "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"
ROUTE_BOUNDARY = "resolve_stage26_7h_boundary_rejections"

ROUTE_FROM_STAGE26_7G = "repair_stage26_7_credit_post_update_eval_binding"
STAGE26_3_SUCCESS_ROUTE = "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"
SUCCESS_METRIC_MAIN_COVERABLE_EFFICIENCY = "main_coverable_coverage_efficiency/v1"
BOUNDARY_FIELDS = ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.7H credit post-update eval binding repair.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_7h_repair_credit_post_update_eval_binding(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_7h_repair_credit_post_update_eval_binding(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_7g_root = Path(config["stage26_7g_root"])
    stage26_7g_summary = _read_json_if_exists(stage26_7g_root / "xunce-stage26-7g-summary.json")
    sweep_rows = _read_jsonl_if_exists(stage26_7g_root / "xunce-stage26-7g-update-sweep-results.jsonl")
    boundary_rejections = _boundary_rejections(config, stage26_7g_summary)
    best_combo = _best_combo(stage26_7g_summary, sweep_rows)
    stage26_2_root = _stage26_2_root(stage26_7g_root, best_combo)
    platform_contract_hash, platform_hash_rejections = _platform_contract_hash_from_stage26_2_root(stage26_2_root)
    input_rejections = _input_rejections(stage26_7g_summary, best_combo, stage26_2_root)
    input_rejections.extend(platform_hash_rejections)

    stage26_3_config: dict[str, Any] = {}
    stage26_3_summary: dict[str, Any] = {}
    if not boundary_rejections and not input_rejections:
        stage26_3_config = _stage26_3_config(config, stage26_2_root, platform_contract_hash=platform_contract_hash)
        _write_json(output_root / STAGE26_3_CONFIG_FILE, stage26_3_config)
        stage26_3_summary = stage26_3.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
            config_path=output_root / STAGE26_3_CONFIG_FILE,
            output_root=output_root / "s26_3_repaired",
            repo_root=repo_root,
        )
    inference_audit = _inference_binding_audit(stage26_3_summary, output_root / "s26_3_repaired")
    unreachable_audit = _unreachable_selected_audit(stage26_3_summary)
    efficiency_audit = _coverage_efficiency_audit(stage26_3_summary)
    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        stage26_3_summary=stage26_3_summary,
        inference_audit=inference_audit,
        unreachable_audit=unreachable_audit,
        efficiency_audit=efficiency_audit,
    )
    status = "passed" if route == ROUTE_MULTI_SEED else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "stage26_7g_root": str(stage26_7g_root),
        "stage26_7g_status": stage26_7g_summary.get("status"),
        "stage26_7g_next_required_change": stage26_7g_summary.get("next_required_change"),
        "best_stable_combo_id": best_combo.get("combo_id"),
        "best_combo_work_dir": best_combo.get("work_dir") or "u0",
        "stage26_2_root": str(stage26_2_root) if stage26_2_root else None,
        "stage26_3_root": str(output_root / "s26_3_repaired"),
        "stage26_3_status": stage26_3_summary.get("status"),
        "stage26_3_next_required_change": stage26_3_summary.get("next_required_change"),
        "post_update_success_metric": stage26_3_summary.get("post_update_success_metric"),
        "stage26_3_handoff_rejections": _stage26_3_efficiency_handoff_rejections(stage26_3_summary),
        "strong_state_join_available_count": int(stage26_3_summary.get("strong_state_join_available_count") or 0),
        "synthetic_inference_required_field_missing_count": inference_audit["synthetic_inference_required_field_missing_count"],
        "hybrid_path_missing_provenance_count": inference_audit["hybrid_path_missing_provenance_count"],
        "hybrid_path_contract_mismatch_count": inference_audit["hybrid_path_contract_mismatch_count"],
        "explicit_unreachable_selected_provenance_count": unreachable_audit["explicit_unreachable_selected_provenance_count"],
        "pre_unreachable_selected_count": unreachable_audit["pre_unreachable_selected_count"],
        "post_unreachable_selected_count": unreachable_audit["post_unreachable_selected_count"],
        "hard_risk_violation_count": int(stage26_3_summary.get("hard_risk_violation_count") or 0),
        "mask_violation_count": int(stage26_3_summary.get("mask_violation_count") or 0),
        "path_planning_failure_count": int(stage26_3_summary.get("path_planning_failure_count") or 0),
        "open_grid_fallback_count": int(stage26_3_summary.get("open_grid_fallback_count") or 0),
        "main_final_coverage_delta": efficiency_audit["main_final_coverage_delta"],
        "main_coverage_auc_delta": efficiency_audit["main_coverage_auc_delta"],
        "main_coverage_per_100m_delta": efficiency_audit["main_coverage_per_100m_delta"],
        "hybrid_astar_path_cost_delta": efficiency_audit["hybrid_astar_path_cost_delta"],
        "hybrid_astar_path_cost_delta_is_diagnostic_only": True,
        "coverage_denominator_source": "main_coverable_cells/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "synthetic_terrain_hash": stage26_3_summary.get("synthetic_terrain_hash"),
        "action_space_type": stage26_3_summary.get("action_space_type") or stage26_3_config.get("action_space_type"),
        "platform_contract_hash": (
            stage26_3_summary.get("platform_contract_hash")
            or stage26_3_config.get("platform_contract_hash")
            or platform_contract_hash
        ),
        "max_traversable_slope_deg": 30.0,
        "input_rejections": input_rejections,
        "boundary_rejections": boundary_rejections,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "input_rejections": input_rejections,
        "boundary_rejections": boundary_rejections,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "summary": str(output_root / SUMMARY_FILE),
        "stage26_3_config": str(output_root / STAGE26_3_CONFIG_FILE),
        "rerun_stage26_3_summary": str(output_root / STAGE26_3_SUMMARY_FILE),
        "inference_binding_audit": str(output_root / INFERENCE_AUDIT_FILE),
        "unreachable_selected_audit": str(output_root / UNREACHABLE_AUDIT_FILE),
        "coverage_efficiency_audit": str(output_root / EFFICIENCY_AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / STAGE26_3_SUMMARY_FILE, stage26_3_summary)
    _write_json(output_root / INFERENCE_AUDIT_FILE, inference_audit)
    _write_json(output_root / UNREACHABLE_AUDIT_FILE, unreachable_audit)
    _write_json(output_root / EFFICIENCY_AUDIT_FILE, efficiency_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _stage26_3_config(config: dict[str, Any], stage26_2_root: Path, *, platform_contract_hash: str | None = None) -> dict[str, Any]:
    base = _read_json(Path(config["stage26_3_base_config"]))
    base.update(
        {
            "stage26_2_root": str(stage26_2_root),
            "stage21_5_base_config": config["stage21_5_base_config"],
            "high_fidelity_config": config["high_fidelity_config"],
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["eval_rollout_steps"]),
            "coverage_denominator_mode": "main_coverable_cells",
            "coverage_denominator_source": "main_coverable_cells/v1",
            "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "hybrid_astar_pose_path_cost_enabled": True,
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
            "continuous_theta_action_space_enabled": True,
            "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
            "default_astar_replaced": False,
            "hybrid_astar_ackermann_feasible_claimed": False,
            "platform_contract_hash": platform_contract_hash or base.get("platform_contract_hash"),
            "stage26_3_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return base


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    stage26_3_summary: dict[str, Any],
    inference_audit: dict[str, Any],
    unreachable_audit: dict[str, Any],
    efficiency_audit: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if (
        inference_audit["synthetic_inference_required_field_missing_count"] > 0
        or inference_audit["hybrid_path_missing_provenance_count"] > 0
        or inference_audit["hybrid_path_contract_mismatch_count"] > 0
    ):
        return ROUTE_BINDING
    stage21_5_route = str(stage26_3_summary.get("stage21_5_next_required_change") or "")
    if unreachable_audit["post_unreachable_selected_count"] > 0 or (
        stage21_5_route == "repair_stage21_5_post_policy_unreachable_regression"
        and unreachable_audit["post_unreachable_selected_count"] > 0
    ):
        return ROUTE_POST_UNREACHABLE
    if unreachable_audit["pre_unreachable_selected_count"] > 0 or (
        stage21_5_route == "repair_stage21_5_pre_policy_unreachable_baseline"
        and unreachable_audit["pre_unreachable_selected_count"] > 0
    ):
        return ROUTE_PRE_UNREACHABLE
    if stage26_3_summary.get("stage21_5_execution_incomplete") is True and stage21_5_route not in {
        "repair_stage21_5_pre_policy_unreachable_baseline",
        "repair_stage21_5_post_policy_unreachable_regression",
    }:
        return ROUTE_STAGE21_5
    selection_changed = (
        int(stage26_3_summary.get("selected_action_changed_count") or 0) > 0
        or int(stage26_3_summary.get("selected_viewpoint_changed_count") or 0) > 0
        or int(stage26_3_summary.get("selected_theta_changed_count") or 0) > 0
    )
    if not selection_changed:
        return ROUTE_MARGIN
    per100 = efficiency_audit["main_coverage_per_100m_delta"]
    if per100 is not None and float(per100) < 0.0:
        return ROUTE_CREDIT
    if (
        float(efficiency_audit["main_final_coverage_delta"] or 0.0) > 0.0
        and per100 is not None
        and float(per100) >= 0.0
        and int(stage26_3_summary.get("scenario_regression_count") or 0) == 0
    ):
        if _stage26_3_efficiency_handoff_rejections(stage26_3_summary):
            return ROUTE_STAGE21_5
        return ROUTE_MULTI_SEED
    return ROUTE_CREDIT


def _stage26_3_efficiency_handoff_rejections(stage26_3_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if stage26_3_summary.get("status") != "passed":
        reasons.append("stage26_3_status_not_passed")
    if stage26_3_summary.get("next_required_change") != STAGE26_3_SUCCESS_ROUTE:
        reasons.append("stage26_3_route_mismatch")
    if stage26_3_summary.get("post_update_success_metric") != SUCCESS_METRIC_MAIN_COVERABLE_EFFICIENCY:
        reasons.append("stage26_3_post_update_success_metric_mismatch")
    return reasons


def _inference_binding_audit(stage26_3_summary: dict[str, Any], stage26_3_root: Path) -> dict[str, Any]:
    action_audit = _read_json_if_exists(stage26_3_root / stage26_3.ACTION_AUDIT_FILE)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "stage26_3_root": str(stage26_3_root),
        "strong_state_join_available_count": int(stage26_3_summary.get("strong_state_join_available_count") or 0),
        "synthetic_inference_required_field_missing_count": int(
            stage26_3_summary.get("synthetic_inference_required_field_missing_count")
            or action_audit.get("synthetic_inference_required_field_missing_count")
            or 0
        ),
        "hybrid_path_missing_provenance_count": int(
            stage26_3_summary.get("hybrid_path_missing_provenance_count")
            or action_audit.get("hybrid_path_missing_provenance_count")
            or 0
        ),
        "hybrid_path_contract_mismatch_count": int(
            stage26_3_summary.get("hybrid_path_contract_mismatch_count")
            or action_audit.get("hybrid_path_contract_mismatch_count")
            or 0
        ),
        "coverage_source_mismatch_count": int(stage26_3_summary.get("coverage_source_mismatch_count") or 0),
        "path_cost_source_mismatch_count": int(stage26_3_summary.get("path_cost_source_mismatch_count") or 0),
        "grid_fallback_count": int(stage26_3_summary.get("grid_fallback_count") or 0),
    }


def _unreachable_selected_audit(stage26_3_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": UNREACHABLE_AUDIT_SCHEMA_VERSION,
        "pre_unreachable_selected_count": int(stage26_3_summary.get("pre_unreachable_selected_count") or 0),
        "post_unreachable_selected_count": int(stage26_3_summary.get("post_unreachable_selected_count") or 0),
        "explicit_unreachable_selected_provenance_count": int(
            stage26_3_summary.get("explicit_unreachable_selected_provenance_count") or 0
        ),
        "stage21_5_next_required_change": stage26_3_summary.get("stage21_5_next_required_change"),
    }


def _coverage_efficiency_audit(stage26_3_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": EFFICIENCY_AUDIT_SCHEMA_VERSION,
        "main_final_coverage_delta": _first_present(stage26_3_summary, "main_final_coverage_delta", "final_coverage_delta"),
        "main_coverage_auc_delta": _first_present(stage26_3_summary, "main_coverage_auc_delta", "coverage_auc_delta"),
        "main_coverage_per_100m_delta": _first_present(stage26_3_summary, "main_coverage_per_100m_delta", "coverage_per_100m_delta"),
        "hybrid_astar_path_cost_delta": _first_present(stage26_3_summary, "hybrid_astar_path_cost_delta", "path_cost_delta"),
        "hybrid_astar_path_cost_delta_is_diagnostic_only": True,
        "scenario_regression_count": int(stage26_3_summary.get("scenario_regression_count") or 0),
    }


def _best_combo(stage26_7g_summary: dict[str, Any], sweep_rows: list[dict[str, Any]]) -> dict[str, Any]:
    desired = str(stage26_7g_summary.get("best_stable_combo_id") or "current_repro")
    for row in sweep_rows:
        if str(row.get("combo_id")) == desired:
            return row
    return {"combo_id": desired, "work_dir": "u0"}


def _stage26_2_root(stage26_7g_root: Path, best_combo: dict[str, Any]) -> Path | None:
    work_dir = str(best_combo.get("work_dir") or "u0")
    root = stage26_7g_root / work_dir / "s26_2"
    return root if root.exists() else None


def _platform_contract_hash_from_stage26_2_root(stage26_2_root: Path | None) -> tuple[str | None, list[str]]:
    if stage26_2_root is None:
        return None, []
    stage26_2_summary = _read_json_if_exists(stage26_2_root / "xunce-stage26-2-summary.json")
    for key in ("platform_contract_hash", "platform_hash"):
        value = _nonempty_string(stage26_2_summary.get(key))
        if value:
            return value, []

    stage26_1_root = _stage26_1_root_from_stage26_2_summary(stage26_2_summary)
    if stage26_1_root is None:
        return None, []
    stage26_1_summary = _read_json_if_exists(stage26_1_root / "xunce-stage26-1-summary.json")
    for key in ("platform_contract_hash", "platform_hash"):
        value = _nonempty_string(stage26_1_summary.get(key))
        if value:
            return value, []

    sidecar_root = stage26_1_root / "src" / "sc"
    if not sidecar_root.exists():
        return None, []
    values: set[str] = set()
    for sidecar_path in sorted(sidecar_root.glob("*.sidecar.json")):
        sidecar = _read_json_if_exists(sidecar_path)
        for key in ("platform_contract_hash", "platform_hash"):
            value = _nonempty_string(sidecar.get(key))
            if value:
                values.add(value)
    if len(values) == 1:
        return next(iter(values)), []
    if len(values) > 1:
        return None, ["stage26_1_sidecar_platform_contract_hash_ambiguous"]
    return None, []


def _stage26_1_root_from_stage26_2_summary(stage26_2_summary: dict[str, Any]) -> Path | None:
    for key in ("stage26_1_root", "stage21_3_ppo_batch_validation_root", "stage21_3_root"):
        value = _nonempty_string(stage26_2_summary.get(key))
        if value:
            root = Path(value)
            if key != "stage26_1_root":
                candidate = root.parent
                if candidate.exists():
                    return candidate
            if root.exists():
                return root
    return None


def _input_rejections(stage26_7g_summary: dict[str, Any], best_combo: dict[str, Any], stage26_2_root: Path | None) -> list[str]:
    reasons: list[str] = []
    if not stage26_7g_summary:
        return ["stage26_7g_summary_missing"]
    if stage26_7g_summary.get("stage_id") != "xunce-stage26-7g-repair-behavior-policy-kl-baseline":
        reasons.append("stage26_7g_wrong_stage_id")
    if stage26_7g_summary.get("next_required_change") != ROUTE_FROM_STAGE26_7G:
        reasons.append("stage26_7g_route_mismatch")
    if stage26_2_root is None:
        reasons.append("best_combo_stage26_2_root_missing")
    if not best_combo:
        reasons.append("best_stable_combo_missing")
    return reasons


def _boundary_rejections(config: dict[str, Any], stage26_7g_summary: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True or stage26_7g_summary.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0 or float(stage26_7g_summary.get("canary_traffic_fraction") or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    config = dict(payload)
    config["stage26_7g_root"] = str(_resolve_path(Path(config.get("stage26_7g_root") or DEFAULT_STAGE26_7G_ROOT), repo_root))
    config["stage26_3_base_config"] = str(_resolve_path(Path(config.get("stage26_3_base_config") or stage26_3.DEFAULT_CONFIG), repo_root))
    base26_3 = _read_json(Path(config["stage26_3_base_config"]))
    config["stage21_5_base_config"] = str(_resolve_path(Path(config.get("stage21_5_base_config") or base26_3["stage21_5_base_config"]), repo_root))
    config["high_fidelity_config"] = str(_resolve_path(Path(config.get("high_fidelity_config") or base26_3["high_fidelity_config"]), repo_root))
    config["required_scenario_count"] = int(config.get("required_scenario_count", 2))
    config["eval_rollout_steps"] = int(config.get("eval_rollout_steps", 4))
    config["hybrid_astar_candidate_eval_workers"] = int(config.get("hybrid_astar_candidate_eval_workers", 4))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    return config


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload.get(key)
    return None


def _nonempty_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.7H Repair Credit Post-Update Eval Binding",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- best_stable_combo_id: `{summary.get('best_stable_combo_id')}`",
            f"- synthetic_inference_required_field_missing_count: `{summary.get('synthetic_inference_required_field_missing_count')}`",
            f"- hybrid_path_missing_provenance_count: `{summary.get('hybrid_path_missing_provenance_count')}`",
            f"- pre_unreachable_selected_count: `{summary.get('pre_unreachable_selected_count')}`",
            f"- post_unreachable_selected_count: `{summary.get('post_unreachable_selected_count')}`",
            f"- main_coverage_per_100m_delta: `{summary.get('main_coverage_per_100m_delta')}`",
            "",
            "Hybrid A* path-cost delta is diagnostic only. This stage does not publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
