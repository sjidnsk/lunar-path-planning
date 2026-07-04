from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import xunce_artifact_io as artifact_io

try:  # pragma: no cover
    import run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1


STAGE_ID = "xunce-stage26-8p-hybrid-astar-primitive-resolution-sweep"
CONFIG_SCHEMA_VERSION = "xunce-stage26-8p-hybrid-astar-primitive-resolution-sweep-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-8p-summary/v1"
SWEEP_ROW_SCHEMA_VERSION = "xunce-stage26-8p-sweep-row/v1"
AUDIT_SCHEMA_VERSION = "xunce-stage26-8p-reachability-resolution-audit/v1"
RUNTIME_SCHEMA_VERSION = "xunce-stage26-8p-runtime-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-8p-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-8p-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep_v1.json"
DEFAULT_OUTPUT_ROOT = "D:/xunce/out/s26_8p"
DEFAULT_STAGE26_8O_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8o_repair_aggressive_collector_trainable_sample_budget_v1"
)
DEFAULT_STAGE26_8G_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8g_repair_synthetic_scenario_diversity_v1"
)

SUMMARY_FILE = "xunce-stage26-8p-summary.json"
SWEEP_RESULTS_FILE = "xunce-stage26-8p-sweep-results.jsonl"
REACHABILITY_AUDIT_FILE = "xunce-stage26-8p-reachability-resolution-audit.json"
RUNTIME_AUDIT_FILE = "xunce-stage26-8p-runtime-audit.json"
RECOMMENDED_STAGE26_8N_CONFIG_FILE = "xunce-stage26-8p-recommended-stage26-8n-config.json"
ROUTING_FILE = "xunce-stage26-8p-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-8p-report.md"
MANIFEST_FILE = "xunce-stage26-8p-manifest.json"

STAGE26_8O_REQUIRED_ROUTE = "expand_stage26_8o_sample_budget_or_start_pool"
ROUTE_INPUTS = "rerun_stage26_8p_required_inputs"
ROUTE_CONTINUE = "continue_stage26_8p_primitive_resolution_sweep"
ROUTE_COLLECTOR = "repair_stage26_8p_collector_binding_or_safety"
ROUTE_RERUN_8N = "rerun_stage26_8n_aggressive_update_sweep_with_repaired_primitive_resolution"
ROUTE_EXPAND_WITH_PRIMITIVE = "expand_stage26_8o_sample_budget_with_repaired_primitive_resolution"
ROUTE_START_POOL = "repair_stage26_synthetic_start_pool_or_candidate_reachability"
ROUTE_BOUNDARY = "resolve_stage26_8p_boundary_rejections"

BOUNDARY_FIELDS = (
    "release_or_training_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.8P Hybrid A* primitive resolution sweep.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-mode", choices=("run_next", "aggregate_only"), default=None)
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
        run_mode_override=args.run_mode,
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    run_mode_override: str | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    if run_mode_override:
        config["run_mode"] = run_mode_override
    output_root = _resolve_path(output_root, repo_root)
    artifact_io.make_dirs(output_root)

    stage26_8o_summary = _read_json_if_exists(Path(config["stage26_8o_root"]) / "xunce-stage26-8o-summary.json")
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(config, stage26_8o_summary)

    if config["run_mode"] == "run_next" and not boundary_rejections and not input_rejections:
        pending = _first_pending_combo(config, output_root)
        if pending:
            _run_combo(config, pending, output_root, repo_root)

    rows = _collect_sweep_rows(config, output_root)
    reachability_audit = _reachability_resolution_audit(config, rows)
    runtime_audit = _runtime_audit(rows)
    route = _route(
        config=config,
        rows=rows,
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        reachability_audit=reachability_audit,
    )
    status = "passed" if route in {ROUTE_CONTINUE, ROUTE_RERUN_8N, ROUTE_EXPAND_WITH_PRIMITIVE, ROUTE_START_POOL} else "failed"
    recommended = _recommended_stage26_8n_config(config, reachability_audit)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "run_mode": config["run_mode"],
        "stage26_8o_root": config["stage26_8o_root"],
        "stage26_8o_status": stage26_8o_summary.get("status"),
        "stage26_8o_next_required_change": stage26_8o_summary.get("next_required_change"),
        "completed_combo_count": len([row for row in rows if row.get("combo_status") == "complete"]),
        "total_combo_count": len(config["primitive_sweep_combos"]),
        "best_combo_id": reachability_audit.get("best_combo_id"),
        "best_primitive_duration_s": reachability_audit.get("best_primitive_duration_s"),
        "best_trainable_transition_count": reachability_audit.get("best_trainable_transition_count", 0),
        "baseline_trainable_transition_count": reachability_audit.get("baseline_trainable_transition_count", 0),
        "trainable_improvement_over_baseline": reachability_audit.get("trainable_improvement_over_baseline", 0),
        "best_no_hybrid_reachable_candidate_terminal_count": reachability_audit.get(
            "best_no_hybrid_reachable_candidate_terminal_count", 0
        ),
        "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
        "coverage_source": config["coverage_source"],
        "path_cost_source": config["path_cost_source"],
        "synthetic_source_kind": config["synthetic_source_kind"],
        "action_space_type": config["action_space_type"],
        "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
        "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "release_or_training_authorized": False,
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
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "summary": str(output_root / SUMMARY_FILE),
        "sweep_results": str(output_root / SWEEP_RESULTS_FILE),
        "reachability_resolution_audit": str(output_root / REACHABILITY_AUDIT_FILE),
        "runtime_audit": str(output_root / RUNTIME_AUDIT_FILE),
        "recommended_stage26_8n_config": str(output_root / RECOMMENDED_STAGE26_8N_CONFIG_FILE),
        "routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_jsonl(output_root / SWEEP_RESULTS_FILE, rows)
    _write_json(output_root / REACHABILITY_AUDIT_FILE, reachability_audit)
    _write_json(output_root / RUNTIME_AUDIT_FILE, runtime_audit)
    _write_json(output_root / RECOMMENDED_STAGE26_8N_CONFIG_FILE, recommended)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    artifact_io.write_text(output_root / REPORT_FILE, _render_report(summary))
    return summary


def _run_combo(config: dict[str, Any], combo: dict[str, Any], output_root: Path, repo_root: Path) -> None:
    combo_root = _combo_root(output_root, combo["combo_id"])
    artifact_io.make_dirs(combo_root)
    stage26_1_config = _build_stage26_1_config(config, combo, repo_root, combo_root)
    config_path = combo_root / "xunce-stage26-8p-stage26-1-config.json"
    _write_json(config_path, stage26_1_config)
    started = time.perf_counter()
    summary = stage26_1.run_xunce_stage26_1_synthetic_terrain_collector_smoke(
        config_path=config_path,
        output_root=combo_root / "s26_1",
        repo_root=repo_root,
    )
    duration_s = time.perf_counter() - started
    _write_json(combo_root / "xunce-stage26-8p-stage26-1-summary.json", summary)
    _write_json(combo_root / "xunce-stage26-8p-combo-runtime.json", {"duration_s": duration_s})


def _build_stage26_1_config(config: dict[str, Any], combo: dict[str, Any], repo_root: Path, combo_root: Path) -> dict[str, Any]:
    cfg = _read_json(_resolve_path(Path(config["base_stage26_1_config"]), repo_root))
    stage21_1_base = _read_json_if_exists(_resolve_path(Path(str(cfg.get("stage21_1_base_config", ""))), repo_root))
    if stage21_1_base:
        stage21_1_base["sampling_seed"] = int(config["sampling_seed"])
        stage21_1_base["continuous_theta_head_init_seed"] = int(config["sampling_seed"])
        generated = combo_root / "_generated_config" / "stage21-1-base-config.json"
        _write_json(generated, stage21_1_base)
        cfg["stage21_1_base_config"] = str(generated)
    cfg.update(
        {
            "stage26_8p_combo_id": combo["combo_id"],
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
            "action_space_type": config["action_space_type"],
            "continuous_theta_action_space_enabled": True,
            "synthetic_credit_feature_exposure_enabled": True,
            "synthetic_exploration_credit_enabled": True,
            "allow_synthetic_credit_behavior_policy": True,
            "selected_continuous_theta_reachability_guard_enabled": True,
            "selected_continuous_theta_unreachable_resample_policy": "reachable_theta_proposal/v1",
            "scenario_diversity_contract_enabled": True,
            "scenario_diversity_source": "synthetic_roi_start_seed_matrix/v1",
            "source_scenario_fixture_root": config["source_scenario_fixture_root"],
            "stage26_8p_source_scenario_fixture_root": config["source_scenario_fixture_root"],
            "scenario_seed_base": int(config["scenario_seed_base"]),
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
            "hybrid_astar_primitive_duration_s": float(combo["hybrid_astar_primitive_duration_s"]),
            "hybrid_astar_integration_dt_s": float(combo["hybrid_astar_integration_dt_s"]),
            "hybrid_astar_max_speed_mps": float(combo["hybrid_astar_max_speed_mps"]),
            "hybrid_astar_max_angular_speed_degps": float(combo["hybrid_astar_max_angular_speed_degps"]),
            "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
            "stage26_1_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _collect_sweep_rows(config: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for combo in config["primitive_sweep_combos"]:
        combo_id = combo["combo_id"]
        combo_root = _combo_root(output_root, combo_id)
        stage26_1_root = combo_root / "s26_1"
        summary = _read_json_if_exists(combo_root / "xunce-stage26-8p-stage26-1-summary.json")
        if not summary:
            summary = _read_json_if_exists(stage26_1_root / stage26_1.SUMMARY_FILE)
        runtime = _read_json_if_exists(combo_root / "xunce-stage26-8p-combo-runtime.json")
        trainable = _read_jsonl_if_exists(stage26_1_root / "s21_1" / stage26_1.stage21_1.TRAINABLE_BATCH_FILE)
        rejections = _read_jsonl_if_exists(stage26_1_root / "s21_1" / stage26_1.stage21_1.REJECTION_FILE)
        reachable_counts = _reachable_counts(trainable, rejections)
        terminal_rows = [row for row in rejections if row.get("reason") == "no_hybrid_reachable_candidate_terminal"]
        row = {
            "schema_version": SWEEP_ROW_SCHEMA_VERSION,
            "combo_id": combo_id,
            "combo_status": "complete" if summary else "pending",
            "combo_root": str(combo_root),
            "stage26_1_root": str(stage26_1_root),
            "stage26_1_status": summary.get("status"),
            "stage26_1_next_required_change": summary.get("next_required_change"),
            "hybrid_astar_primitive_duration_s": float(combo["hybrid_astar_primitive_duration_s"]),
            "hybrid_astar_integration_dt_s": float(combo["hybrid_astar_integration_dt_s"]),
            "hybrid_astar_max_speed_mps": float(combo["hybrid_astar_max_speed_mps"]),
            "hybrid_astar_max_angular_speed_degps": float(combo["hybrid_astar_max_angular_speed_degps"]),
            "trainable_transition_count": int(summary.get("trainable_transition_count") or summary.get("transition_count") or summary.get("batch_row_count") or 0),
            "trainable_transition_count_by_scenario": summary.get("stage21_1_trainable_transition_count_by_scenario") or {},
            "no_hybrid_reachable_candidate_terminal_count": int(
                summary.get("stage21_1_no_hybrid_reachable_candidate_terminal_count")
                or summary.get("rejection_no_hybrid_reachable_candidate_terminal_count")
                or len(terminal_rows)
            ),
            "selected_continuous_theta_unreachable_attempt_count": int(
                summary.get("stage21_1_selected_continuous_theta_unreachable_attempt_count") or 0
            ),
            "selected_continuous_theta_resample_success_count": int(
                summary.get("stage21_1_selected_continuous_theta_resample_success_count") or 0
            ),
            "selected_pose_unreachable_terminal_count": int(
                summary.get("stage21_1_selected_pose_unreachable_terminal_count")
                or summary.get("rejection_no_selected_reachable_pose_candidate_terminal_count")
                or 0
            ),
            "reward_row_count": int(summary.get("reward_row_count") or 0),
            "batch_row_count": int(summary.get("batch_row_count") or 0),
            "mean_hybrid_astar_reachable_count_per_step": _mean(reachable_counts),
            "min_hybrid_astar_reachable_count_per_step": min(reachable_counts) if reachable_counts else None,
            "max_hybrid_astar_reachable_count_per_step": max(reachable_counts) if reachable_counts else None,
            "terminal_rows": _terminal_row_audit(terminal_rows),
            "terminal_row_count": len(terminal_rows),
            "runtime_seconds": _finite(runtime.get("duration_s")),
            "collector_binding_or_safety_failure": _collector_binding_or_safety_failure(summary),
            "coverage_source": summary.get("coverage_source") or config["coverage_source"],
            "path_cost_source": summary.get("path_cost_source") or config["path_cost_source"],
            "synthetic_source_kind": summary.get("synthetic_source_kind") or config["synthetic_source_kind"],
            "synthetic_terrain_hash": summary.get("synthetic_terrain_hash"),
            "platform_contract_hash": summary.get("platform_contract_hash"),
            "release_or_training_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
        rows.append(row)
    return rows


def _reachable_counts(trainable: list[dict[str, Any]], rejections: list[dict[str, Any]]) -> list[int]:
    counts: list[int] = []
    for row in trainable:
        info = row.get("info") if isinstance(row.get("info"), dict) else row
        flags = info.get("hybrid_astar_reachable_flags") if isinstance(info, dict) else None
        if isinstance(flags, list):
            counts.append(sum(1 for value in flags if value is True))
    for row in rejections:
        value = row.get("hybrid_astar_reachable_count")
        if isinstance(value, int):
            counts.append(value)
    return counts


def _terminal_row_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": row.get("scenario_id"),
            "step_index": row.get("step_index"),
            "action_mask_true_count": row.get("action_mask_true_count"),
            "hard_risk_clean_mask_true_count": row.get("hard_risk_clean_mask_true_count"),
            "hybrid_astar_reachable_count": row.get("hybrid_astar_reachable_count"),
            "sampling_mask_true_count": row.get("sampling_mask_true_count"),
        }
        for row in rows
    ]


def _reachability_resolution_audit(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["combo_status"] == "complete"]
    baseline = next((row for row in rows if row["combo_id"] == config["baseline_combo_id"] and row["combo_status"] == "complete"), {})
    best = max(
        complete,
        key=lambda row: (
            int(row.get("trainable_transition_count") or 0),
            -int(row.get("no_hybrid_reachable_candidate_terminal_count") or 0),
            -float(row.get("hybrid_astar_primitive_duration_s") or 0.0),
        ),
        default={},
    )
    baseline_count = int(baseline.get("trainable_transition_count") or 0)
    best_count = int(best.get("trainable_transition_count") or 0)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "completed_combo_count": len(complete),
        "total_combo_count": len(rows),
        "baseline_combo_id": config["baseline_combo_id"],
        "baseline_trainable_transition_count": baseline_count,
        "best_combo_id": best.get("combo_id"),
        "best_primitive_duration_s": best.get("hybrid_astar_primitive_duration_s"),
        "best_trainable_transition_count": best_count,
        "best_no_hybrid_reachable_candidate_terminal_count": int(best.get("no_hybrid_reachable_candidate_terminal_count") or 0),
        "trainable_improvement_over_baseline": best_count - baseline_count,
        "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
        "target_met": best_count >= int(config["min_trainable_transition_count"]),
        "material_reachability_improvement": (best_count - baseline_count) >= int(config["material_trainable_improvement_threshold"]),
        "primitive_resolution_appears_material": best_count >= int(config["min_trainable_transition_count"])
        or (best_count - baseline_count) >= int(config["material_trainable_improvement_threshold"]),
    }


def _runtime_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "runtime_seconds_by_combo": {
            str(row["combo_id"]): row.get("runtime_seconds") for row in rows if row.get("runtime_seconds") is not None
        },
        "runtime_is_diagnostic_only": True,
    }


def _route(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    boundary_rejections: list[str],
    input_rejections: list[str],
    reachability_audit: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    complete = [row for row in rows if row["combo_status"] == "complete"]
    if len(complete) < len(config["primitive_sweep_combos"]):
        return ROUTE_CONTINUE
    if any(row.get("collector_binding_or_safety_failure") for row in complete):
        return ROUTE_COLLECTOR
    if reachability_audit.get("target_met") is True:
        return ROUTE_RERUN_8N
    if reachability_audit.get("material_reachability_improvement") is True:
        return ROUTE_EXPAND_WITH_PRIMITIVE
    return ROUTE_START_POOL


def _recommended_stage26_8n_config(config: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    combo = next((item for item in config["primitive_sweep_combos"] if item["combo_id"] == audit.get("best_combo_id")), {})
    return {
        "schema_version": "xunce-stage26-8p-recommended-stage26-8n-config/v1",
        "recommended_combo_id": audit.get("best_combo_id"),
        "hybrid_astar_primitive_duration_s": combo.get("hybrid_astar_primitive_duration_s"),
        "hybrid_astar_integration_dt_s": combo.get("hybrid_astar_integration_dt_s"),
        "hybrid_astar_max_speed_mps": combo.get("hybrid_astar_max_speed_mps"),
        "hybrid_astar_max_angular_speed_degps": combo.get("hybrid_astar_max_angular_speed_degps"),
        "selected_continuous_theta_reachability_guard_enabled": True,
        "selected_continuous_theta_unreachable_resample_policy": "reachable_theta_proposal/v1",
        "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
        "coverage_source": config["coverage_source"],
        "path_cost_source": config["path_cost_source"],
        "synthetic_source_kind": config["synthetic_source_kind"],
        "action_space_type": config["action_space_type"],
        "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
        "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _first_pending_combo(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    for combo in config["primitive_sweep_combos"]:
        summary_path = _combo_root(output_root, combo["combo_id"]) / "xunce-stage26-8p-stage26-1-summary.json"
        stage_summary_path = _combo_root(output_root, combo["combo_id"]) / "s26_1" / stage26_1.SUMMARY_FILE
        if not artifact_io.path_is_file(summary_path) and not artifact_io.path_is_file(stage_summary_path):
            return combo
    return {}


def _combo_root(output_root: Path, combo_id: str) -> Path:
    return output_root / "combos" / combo_id


def _collector_binding_or_safety_failure(summary: dict[str, Any]) -> bool:
    if not summary:
        return False
    safety_fields = (
        "hard_risk_violation_count",
        "mask_violation_count",
        "path_planning_failure_count",
        "open_grid_fallback_count",
        "synthetic_physical_obstacle_pollution_count",
    )
    return any(int(summary.get(field) or 0) > 0 for field in safety_fields) or summary.get(
        "stage21_1_source_roi_expansion_root_match"
    ) is False


def _input_rejections(config: dict[str, Any], stage26_8o_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not stage26_8o_summary:
        reasons.append("missing_stage26_8o_summary")
    elif stage26_8o_summary.get("next_required_change") != STAGE26_8O_REQUIRED_ROUTE:
        reasons.append("stage26_8o_route_not_sample_budget_or_start_pool")
    if not artifact_io.path_exists(Path(config["source_scenario_fixture_root"])):
        reasons.append("source_scenario_fixture_root_missing")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if bool(config.get(field))]
    if float(config.get("canary_traffic_fraction") or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    config = _default_config() | payload
    for key in ("stage26_8o_root", "source_scenario_fixture_root", "base_stage26_1_config"):
        config[key] = str(_resolve_path(Path(str(config[key])), repo_root))
    config["run_mode"] = str(config.get("run_mode") or "run_next")
    if config["run_mode"] not in {"run_next", "aggregate_only"}:
        raise ValueError("run_mode must be run_next or aggregate_only")
    config["required_scenario_count"] = _positive_int(config["required_scenario_count"], "required_scenario_count")
    config["rollout_steps"] = _positive_int(config["rollout_steps"], "rollout_steps")
    config["min_trainable_transition_count"] = _positive_int(
        config["min_trainable_transition_count"], "min_trainable_transition_count"
    )
    config["material_trainable_improvement_threshold"] = _positive_int(
        config.get("material_trainable_improvement_threshold", 20),
        "material_trainable_improvement_threshold",
    )
    config["sampling_seed"] = _nonnegative_int(config["sampling_seed"], "sampling_seed")
    config["scenario_seed_base"] = _nonnegative_int(config["scenario_seed_base"], "scenario_seed_base")
    config["hybrid_astar_candidate_eval_workers"] = _positive_int(
        config["hybrid_astar_candidate_eval_workers"], "hybrid_astar_candidate_eval_workers"
    )
    config["max_traversable_slope_deg"] = _positive_float(config["max_traversable_slope_deg"], "max_traversable_slope_deg")
    config["primitive_sweep_combos"] = _primitive_sweep_combos(config["primitive_sweep_combos"])
    combo_ids = {combo["combo_id"] for combo in config["primitive_sweep_combos"]}
    if config["baseline_combo_id"] not in combo_ids:
        raise ValueError("baseline_combo_id must name a primitive_sweep_combos combo_id")
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    return config


def _default_config() -> dict[str, Any]:
    return {
        "run_mode": "run_next",
        "stage26_8o_root": DEFAULT_STAGE26_8O_ROOT,
        "source_scenario_fixture_root": DEFAULT_STAGE26_8G_ROOT,
        "base_stage26_1_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
        "required_scenario_count": 6,
        "rollout_steps": 20,
        "min_trainable_transition_count": 100,
        "sampling_seed": 260801,
        "scenario_seed_base": 260801,
        "baseline_combo_id": "p4_0_baseline",
        "material_trainable_improvement_threshold": 20,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "hybrid_astar_candidate_eval_workers": 4,
        "max_traversable_slope_deg": 30.0,
        "primitive_sweep_combos": [
            {
                "combo_id": "p4_0_baseline",
                "hybrid_astar_primitive_duration_s": 4.0,
                "hybrid_astar_integration_dt_s": 0.25,
                "hybrid_astar_max_speed_mps": 1.0,
                "hybrid_astar_max_angular_speed_degps": 45.0,
            },
            {
                "combo_id": "p2_0_medium",
                "hybrid_astar_primitive_duration_s": 2.0,
                "hybrid_astar_integration_dt_s": 0.25,
                "hybrid_astar_max_speed_mps": 1.0,
                "hybrid_astar_max_angular_speed_degps": 45.0,
            },
            {
                "combo_id": "p1_0_fine",
                "hybrid_astar_primitive_duration_s": 1.0,
                "hybrid_astar_integration_dt_s": 0.25,
                "hybrid_astar_max_speed_mps": 1.0,
                "hybrid_astar_max_angular_speed_degps": 45.0,
            },
        ],
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _primitive_sweep_combos(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("primitive_sweep_combos must be a non-empty list")
    combos: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        combo_id = str(item["combo_id"])
        if combo_id in seen:
            raise ValueError(f"duplicate combo_id: {combo_id}")
        seen.add(combo_id)
        combos.append(
            {
                "combo_id": combo_id,
                "hybrid_astar_primitive_duration_s": _positive_float(
                    item["hybrid_astar_primitive_duration_s"], "hybrid_astar_primitive_duration_s"
                ),
                "hybrid_astar_integration_dt_s": _positive_float(
                    item["hybrid_astar_integration_dt_s"], "hybrid_astar_integration_dt_s"
                ),
                "hybrid_astar_max_speed_mps": _positive_float(item["hybrid_astar_max_speed_mps"], "hybrid_astar_max_speed_mps"),
                "hybrid_astar_max_angular_speed_degps": _positive_float(
                    item["hybrid_astar_max_angular_speed_degps"], "hybrid_astar_max_angular_speed_degps"
                ),
            }
        )
    return combos


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.8P Hybrid A* Primitive Resolution Sweep",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- completed_combo_count: `{summary['completed_combo_count']}/{summary['total_combo_count']}`",
            f"- best_combo_id: `{summary.get('best_combo_id')}`",
            f"- best_primitive_duration_s: `{summary.get('best_primitive_duration_s')}`",
            f"- best_trainable_transition_count: `{summary.get('best_trainable_transition_count')}`",
            f"- baseline_trainable_transition_count: `{summary.get('baseline_trainable_transition_count')}`",
            "",
            "This stage only runs collector/reward/batch chain and does not run PPO update/eval.",
        ]
    )


def _mean(values: list[int]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _read_json(path: Path) -> dict[str, Any]:
    return artifact_io.read_json(path)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if artifact_io.path_is_file(path) else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return artifact_io.read_jsonl(path) if artifact_io.path_is_file(path) else []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    artifact_io.write_json(path, payload)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    artifact_io.write_jsonl(path, rows)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive float") from exc
    if parsed <= 0.0 or not math.isfinite(parsed):
        raise ValueError(f"{name} must be a positive finite float")
    return parsed


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
