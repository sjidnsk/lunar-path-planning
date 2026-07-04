from __future__ import annotations

import argparse
import json
import sys
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


STAGE_ID = "xunce-stage26-8o-repair-aggressive-collector-trainable-sample-budget"
CONFIG_SCHEMA_VERSION = "xunce-stage26-8o-repair-aggressive-collector-trainable-sample-budget-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-8o-summary/v1"
AUDIT_SCHEMA_VERSION = "xunce-stage26-8o-reachability-guard-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-8o-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-8o-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_8o_repair_aggressive_collector_trainable_sample_budget_v1.json"
DEFAULT_OUTPUT_ROOT = "D:/xunce/out/s26_8o"
DEFAULT_STAGE26_8N_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8n_aggressive_sample_update_sweep_v1"
)
DEFAULT_STAGE26_8G_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8g_repair_synthetic_scenario_diversity_v1"
)

SUMMARY_FILE = "xunce-stage26-8o-summary.json"
STAGE26_1_CONFIG_FILE = "xunce-stage26-8o-stage26-1-config.json"
STAGE26_1_SUMMARY_FILE = "xunce-stage26-8o-stage26-1-summary.json"
REACHABILITY_AUDIT_FILE = "xunce-stage26-8o-reachability-guard-audit.json"
RECOMMENDED_STAGE26_8N_CONFIG_FILE = "xunce-stage26-8o-recommended-stage26-8n-config.json"
ROUTING_FILE = "xunce-stage26-8o-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-8o-report.md"
MANIFEST_FILE = "xunce-stage26-8o-manifest.json"

STAGE26_8N_REQUIRED_ROUTE = "expand_stage26_8n_aggressive_sample_budget"
ROUTE_INPUTS = "rerun_stage26_8o_required_inputs"
ROUTE_GUARD = "repair_stage26_8o_selected_theta_reachability_guard"
ROUTE_LOGPROB = "repair_stage26_8o_behavior_logprob_contract"
ROUTE_EXPAND = "expand_stage26_8o_sample_budget_or_start_pool"
ROUTE_COLLECTOR = "repair_stage26_8o_collector_binding_or_safety"
ROUTE_RERUN_8N = "rerun_stage26_8n_aggressive_update_sweep_with_repaired_collector"
ROUTE_BOUNDARY = "resolve_stage26_8o_boundary_rejections"

BOUNDARY_FIELDS = (
    "release_or_training_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair aggressive collector trainable sample budget.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_8o_repair_aggressive_collector_trainable_sample_budget(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_8o_repair_aggressive_collector_trainable_sample_budget(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    artifact_io.make_dirs(output_root)
    config["_generated_config_root"] = str(output_root / "_generated_config")

    stage26_8n_summary = _read_json_if_exists(Path(config["stage26_8n_root"]) / "xunce-stage26-8n-summary.json")
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(config, stage26_8n_summary)

    stage26_1_config = _build_stage26_1_config(config, repo_root)
    stage26_1_config_path = output_root / STAGE26_1_CONFIG_FILE
    _write_json(stage26_1_config_path, stage26_1_config)
    stage26_1_root = output_root / "s26_1"
    if not boundary_rejections and not input_rejections and bool(config["run_stage26_1"]):
        stage26_1_summary = stage26_1.run_xunce_stage26_1_synthetic_terrain_collector_smoke(
            config_path=stage26_1_config_path,
            output_root=stage26_1_root,
            repo_root=repo_root,
        )
    else:
        stage26_1_summary = _read_json_if_exists(stage26_1_root / stage26_1.SUMMARY_FILE)

    audit = _reachability_guard_audit(config, stage26_1_summary, stage26_1_root)
    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        stage26_1_summary=stage26_1_summary,
        audit=audit,
        min_trainable_transition_count=int(config["min_trainable_transition_count"]),
    )
    status = "passed" if route == ROUTE_RERUN_8N else "failed"
    recommended = _recommended_stage26_8n_config(config, stage26_1_root)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "stage26_8n_root": config["stage26_8n_root"],
        "stage26_8n_status": stage26_8n_summary.get("status"),
        "stage26_8n_next_required_change": stage26_8n_summary.get("next_required_change"),
        "stage26_1_root": str(stage26_1_root),
        "stage26_1_status": stage26_1_summary.get("status"),
        "stage26_1_next_required_change": stage26_1_summary.get("next_required_change"),
        "trainable_transition_count": audit["trainable_transition_count"],
        "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
        "synthetic_credit_target_selected_count": audit["synthetic_credit_target_selected_count"],
        "selected_continuous_theta_unreachable_attempt_count": audit[
            "selected_continuous_theta_unreachable_attempt_count"
        ],
        "selected_continuous_theta_resample_success_count": audit[
            "selected_continuous_theta_resample_success_count"
        ],
        "selected_candidate_resample_success_count": audit["selected_candidate_resample_success_count"],
        "selected_pose_unreachable_terminal_count": audit["selected_pose_unreachable_terminal_count"],
        "behavior_logprob_recompute_evaluated": audit["behavior_logprob_recompute_evaluated"],
        "behavior_logprob_recomputable": audit["behavior_logprob_recomputable"],
        "collector_binding_or_safety_failure": audit["collector_binding_or_safety_failure"],
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
        "stage26_1_config": str(stage26_1_config_path),
        "stage26_1_summary": str(output_root / STAGE26_1_SUMMARY_FILE),
        "reachability_guard_audit": str(output_root / REACHABILITY_AUDIT_FILE),
        "recommended_stage26_8n_config": str(output_root / RECOMMENDED_STAGE26_8N_CONFIG_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_json(output_root / STAGE26_1_SUMMARY_FILE, stage26_1_summary)
    _write_json(output_root / REACHABILITY_AUDIT_FILE, audit)
    _write_json(output_root / RECOMMENDED_STAGE26_8N_CONFIG_FILE, recommended)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    artifact_io.write_text(output_root / REPORT_FILE, _render_report(summary))
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    config = dict(payload)
    for key in ("stage26_8n_root", "source_scenario_fixture_root", "base_stage26_1_config"):
        if not isinstance(config.get(key), str) or not str(config[key]).strip():
            raise ValueError(f"{key} must be a non-empty path string")
        config[key] = str(_resolve_path(Path(config[key]), repo_root))
    config["run_stage26_1"] = bool(config.get("run_stage26_1", True))
    config["required_scenario_count"] = _positive_int(config.get("required_scenario_count", 6), "required_scenario_count")
    config["rollout_steps"] = _positive_int(config.get("rollout_steps", 20), "rollout_steps")
    config["min_trainable_transition_count"] = _positive_int(
        config.get("min_trainable_transition_count", 100),
        "min_trainable_transition_count",
    )
    config["sampling_seed"] = _nonnegative_int(config.get("sampling_seed", 260801), "sampling_seed")
    config["scenario_seed_base"] = _nonnegative_int(config.get("scenario_seed_base", config["sampling_seed"]), "scenario_seed_base")
    config["hybrid_astar_candidate_eval_workers"] = _positive_int(
        config.get("hybrid_astar_candidate_eval_workers", 4),
        "hybrid_astar_candidate_eval_workers",
    )
    config["max_traversable_slope_deg"] = _positive_float(
        config.get("max_traversable_slope_deg", 30.0),
        "max_traversable_slope_deg",
    )
    config["coverage_source"] = str(config.get("coverage_source") or "endpoint_theta_slope_obstacle_los/v1")
    config["path_cost_source"] = str(config.get("path_cost_source") or "hybrid_astar_pose_path/v1")
    config["synthetic_source_kind"] = str(config.get("synthetic_source_kind") or "synthetic_terrain_obstacle_proxy/v1")
    config["action_space_type"] = str(config.get("action_space_type") or "hybrid_discrete_xy_continuous_theta/v1")
    config["selected_continuous_theta_reachability_guard_enabled"] = bool(
        config.get("selected_continuous_theta_reachability_guard_enabled", True)
    )
    config["selected_continuous_theta_unreachable_resample_policy"] = str(
        config.get("selected_continuous_theta_unreachable_resample_policy") or "reachable_theta_proposal/v1"
    )
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    for field in BOUNDARY_FIELDS:
        config.setdefault(field, False)
    return config


def _build_stage26_1_config(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    cfg = _read_json(_resolve_path(Path(config["base_stage26_1_config"]), repo_root))
    cfg.update(
        {
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
            "action_space_type": config["action_space_type"],
            "continuous_theta_action_space_enabled": True,
            "synthetic_credit_feature_exposure_enabled": True,
            "synthetic_exploration_credit_enabled": True,
            "allow_synthetic_credit_behavior_policy": True,
            "selected_continuous_theta_reachability_guard_enabled": bool(
                config["selected_continuous_theta_reachability_guard_enabled"]
            ),
            "selected_continuous_theta_unreachable_resample_policy": config[
                "selected_continuous_theta_unreachable_resample_policy"
            ],
            "scenario_diversity_contract_enabled": True,
            "scenario_diversity_source": "synthetic_roi_start_seed_matrix/v1",
            "source_scenario_fixture_root": config["source_scenario_fixture_root"],
            "stage26_8o_source_scenario_fixture_root": config["source_scenario_fixture_root"],
            "scenario_seed_base": int(config["scenario_seed_base"]),
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
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
    stage21_1_base_path = _resolve_path(Path(str(cfg.get("stage21_1_base_config", ""))), repo_root)
    stage21_1_base = _read_json_if_exists(stage21_1_base_path)
    if stage21_1_base:
        stage21_1_base["sampling_seed"] = int(config["sampling_seed"])
        stage21_1_base["continuous_theta_head_init_seed"] = int(config["sampling_seed"])
        generated = Path(config.get("_generated_config_root", ".")) / "stage21-1-base-config.json"
        artifact_io.make_dirs(generated.parent)
        _write_json(generated, stage21_1_base)
        cfg["stage21_1_base_config"] = str(generated)
    return cfg


def _reachability_guard_audit(config: dict[str, Any], stage26_1_summary: dict[str, Any], stage26_1_root: Path) -> dict[str, Any]:
    stage21_1_summary = _read_json_if_exists(stage26_1_root / stage26_1.STAGE21_1_SUMMARY_FILE)
    stage21_3_summary = _read_json_if_exists(stage26_1_root / stage26_1.STAGE21_3_SUMMARY_FILE)
    transitions = _read_jsonl_if_exists(stage26_1_root / "s21_1" / stage26_1.stage21_1.TRAINABLE_BATCH_FILE)
    resampled_rows = [
        row
        for row in transitions
        if row.get("selected_theta_resampled_for_reachability") is True
        or row.get("selected_candidate_resampled_for_reachability") is True
    ]
    behavior_rows = [
        row
        for row in transitions
        if row.get("behavior_policy_id") == "continuous_theta_reachability_guard_policy/v1"
        or (isinstance(row.get("info"), dict) and row["info"].get("behavior_policy_id") == "continuous_theta_reachability_guard_policy/v1")
    ]
    stage21_3_evaluated = bool(stage21_3_summary) or stage26_1_summary.get("stage21_3_status") is not None
    behavior_logprob_recomputable = (
        True
        if not behavior_rows
        else stage21_3_summary.get("status") == "passed"
        if stage21_3_evaluated
        else None
    )
    safety_counts = {
        "hard_risk_violation_count": int(stage26_1_summary.get("hard_risk_violation_count") or 0),
        "mask_violation_count": int(stage26_1_summary.get("mask_violation_count") or 0),
        "path_planning_failure_count": int(stage26_1_summary.get("path_planning_failure_count") or 0),
        "open_grid_fallback_count": int(stage26_1_summary.get("open_grid_fallback_count") or 0),
    }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "stage26_1_status": stage26_1_summary.get("status"),
        "stage26_1_next_required_change": stage26_1_summary.get("next_required_change"),
        "stage21_1_status": stage21_1_summary.get("status") or stage26_1_summary.get("stage21_1_status"),
        "stage21_3_status": stage21_3_summary.get("status") or stage26_1_summary.get("stage21_3_status"),
        "trainable_transition_count": int(
            stage21_1_summary.get("trainable_transition_count")
            or stage26_1_summary.get("batch_row_count")
            or stage26_1_summary.get("transition_count")
            or 0
        ),
        "synthetic_credit_target_selected_count": int(
            stage26_1_summary.get("stage21_1_synthetic_credit_target_selected_count_from_transition_rows") or 0
        ),
        "selected_continuous_theta_reachability_guard_enabled": bool(
            config["selected_continuous_theta_reachability_guard_enabled"]
        ),
        "selected_continuous_theta_unreachable_resample_policy": config[
            "selected_continuous_theta_unreachable_resample_policy"
        ],
        "selected_continuous_theta_unreachable_attempt_count": int(
            stage26_1_summary.get("stage21_1_selected_continuous_theta_unreachable_attempt_count") or 0
        ),
        "selected_continuous_theta_resample_success_count": int(
            stage26_1_summary.get("stage21_1_selected_continuous_theta_resample_success_count") or 0
        ),
        "selected_candidate_resample_success_count": int(
            stage26_1_summary.get("stage21_1_selected_candidate_resample_success_count") or 0
        ),
        "selected_pose_unreachable_terminal_count": int(
            stage26_1_summary.get("stage21_1_selected_pose_unreachable_terminal_count") or 0
        ),
        "resampled_behavior_row_count": len(behavior_rows),
        "resampled_transition_row_count": len(resampled_rows),
        "behavior_logprob_recompute_evaluated": bool(stage21_3_evaluated),
        "behavior_logprob_recomputable": behavior_logprob_recomputable,
        "collector_binding_or_safety_failure": any(value > 0 for value in safety_counts.values())
        or stage26_1_summary.get("stage21_1_source_roi_expansion_root_match") is False,
        **safety_counts,
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    stage26_1_summary: dict[str, Any],
    audit: dict[str, Any],
    min_trainable_transition_count: int,
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if not audit["selected_continuous_theta_reachability_guard_enabled"]:
        return ROUTE_GUARD
    if stage26_1_summary.get("stage21_1_status") != "passed" and audit["selected_continuous_theta_unreachable_attempt_count"] <= 0:
        return ROUTE_GUARD
    if audit["collector_binding_or_safety_failure"]:
        return ROUTE_COLLECTOR
    if int(audit["trainable_transition_count"]) < min_trainable_transition_count:
        return ROUTE_EXPAND
    if not audit["behavior_logprob_recomputable"]:
        return ROUTE_LOGPROB
    return ROUTE_RERUN_8N


def _input_rejections(config: dict[str, Any], stage26_8n_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not stage26_8n_summary:
        reasons.append("missing_stage26_8n_summary")
    elif stage26_8n_summary.get("next_required_change") != STAGE26_8N_REQUIRED_ROUTE:
        reasons.append("stage26_8n_route_not_aggressive_sample_budget")
    if not artifact_io.path_exists(Path(config["source_scenario_fixture_root"])):
        reasons.append("source_scenario_fixture_root_missing")
    if config["selected_continuous_theta_unreachable_resample_policy"] != "reachable_theta_proposal/v1":
        reasons.append("unsupported_selected_theta_resample_policy")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in BOUNDARY_FIELDS:
        if config.get(field) is not False:
            reasons.append(field)
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _recommended_stage26_8n_config(config: dict[str, Any], stage26_1_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage26-8n-recommended-repaired-collector-config/v1",
        "repaired_collector_root": str(stage26_1_root),
        "reuse_repaired_collector_root": True,
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


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.8O Repair Aggressive Collector Trainable Sample Budget",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- trainable_transition_count: `{summary['trainable_transition_count']}`",
            f"- min_trainable_transition_count: `{summary['min_trainable_transition_count']}`",
            f"- selected_continuous_theta_unreachable_attempt_count: `{summary['selected_continuous_theta_unreachable_attempt_count']}`",
            f"- selected_continuous_theta_resample_success_count: `{summary['selected_continuous_theta_resample_success_count']}`",
            f"- selected_candidate_resample_success_count: `{summary['selected_candidate_resample_success_count']}`",
            f"- selected_pose_unreachable_terminal_count: `{summary['selected_pose_unreachable_terminal_count']}`",
            f"- behavior_logprob_recomputable: `{summary['behavior_logprob_recomputable']}`",
            "",
            "This stage only repairs collector sampling reachability and does not run PPO update/eval.",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    return artifact_io.read_json(path)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not artifact_io.path_is_file(path):
        return {}
    return _read_json(path)


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return artifact_io.read_jsonl(path) if artifact_io.path_is_file(path) else []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    artifact_io.write_json(path, payload)


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
    if parsed <= 0.0:
        raise ValueError(f"{name} must be a positive float")
    return parsed


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
