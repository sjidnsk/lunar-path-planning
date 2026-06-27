from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
    import run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3


STAGE_ID = "xunce-stage26-6-synthetic-exploration-credit-assignment"
CONFIG_SCHEMA_VERSION = "xunce-stage26-6-synthetic-exploration-credit-assignment-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-6-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-6-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-6-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_6_synthetic_exploration_credit_assignment_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_6_synthetic_exploration_credit_assignment_v1"
)
DEFAULT_STAGE26_5_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration_v1"
)
DEFAULT_STAGE26_0_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract_v1"
)

SYNTHETIC_SOURCE_KIND = "synthetic_terrain_obstacle_proxy/v1"
COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"
ACTION_SPACE_TYPE = "hybrid_discrete_xy_continuous_theta/v1"
ROUTE_FROM_STAGE26_5 = "repair_stage26_synthetic_exploration_credit_assignment"

SUMMARY_FILE = "xunce-stage26-6-summary.json"
TARGET_AUDIT_FILE = "xunce-stage26-6-synthetic-credit-target-audit.jsonl"
BEHAVIOR_AUDIT_FILE = "xunce-stage26-6-behavior-logprob-audit.json"
FEATURE_AUDIT_FILE = "xunce-stage26-6-candidate-feature-exposure-audit.json"
STAGE26_1_SUMMARY_FILE = "xunce-stage26-6-stage26-1-summary.json"
STAGE26_2_SUMMARY_FILE = "xunce-stage26-6-stage26-2-summary.json"
STAGE26_3_SUMMARY_FILE = "xunce-stage26-6-stage26-3-summary.json"
ROUTING_FILE = "xunce-stage26-6-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-6-report.md"
MANIFEST_FILE = "xunce-stage26-6-manifest.json"

ROUTE_INPUTS = "rerun_stage26_6_required_inputs"
ROUTE_FEATURE = "repair_stage26_6_candidate_feature_exposure"
ROUTE_TARGET = "repair_stage26_6_synthetic_credit_target_contract"
ROUTE_SAMPLER = "repair_stage26_6_synthetic_credit_sampler"
ROUTE_LOGPROB = "repair_stage26_6_behavior_logprob_contract"
ROUTE_STABILITY = "repair_stage26_6_credit_ppo_update_stability"
ROUTE_GRADIENT = "repair_stage26_synthetic_policy_gradient_direction"
ROUTE_MARGIN = "calibrate_stage26_synthetic_discrete_margin_crossing_after_credit"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_STAGE26_7 = "run_stage26_7_synthetic_terrain_multi_seed_ppo_pilot"
ROUTE_STAGE26_3_MULTI_SEED_ALIAS = "run_stage26_4_synthetic_terrain_multi_seed_ppo_pilot"
ROUTE_BOUNDARY = "resolve_stage26_6_boundary_rejections"

BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.6 synthetic exploration credit assignment smoke.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_6_synthetic_exploration_credit_assignment(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_6_synthetic_exploration_credit_assignment(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_5_summary = _read_stage26_5_summary(Path(config["stage26_5_root"]))
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(stage26_5_summary)

    stage26_1_summary: dict[str, Any] = {}
    stage26_2_summary: dict[str, Any] = {}
    stage26_3_summary: dict[str, Any] = {}
    feature_audit = _empty_feature_audit()
    behavior_audit = _empty_behavior_audit()
    target_rows: list[dict[str, Any]] = []

    if not boundary_rejections and not input_rejections and bool(config["run_stage26_chain"]):
        chain = _run_stage26_chain(config=config, output_root=output_root, repo_root=repo_root)
        stage26_1_summary = chain["stage26_1_summary"]
        stage26_2_summary = chain["stage26_2_summary"]
        stage26_3_summary = chain["stage26_3_summary"]
        target_rows = chain["target_rows"]
        feature_audit = chain["feature_audit"]
        behavior_audit = chain["behavior_audit"]

    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        config=config,
        feature_audit=feature_audit,
        behavior_audit=behavior_audit,
        target_rows=target_rows,
        stage26_1_summary=stage26_1_summary,
        stage26_2_summary=stage26_2_summary,
        stage26_3_summary=stage26_3_summary,
    )
    status = "passed" if route == ROUTE_STAGE26_7 else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "stage26_5_root": config["stage26_5_root"],
        "stage26_5_status": stage26_5_summary.get("status"),
        "stage26_5_next_required_change": stage26_5_summary.get("next_required_change"),
        "run_stage26_chain": bool(config["run_stage26_chain"]),
        "feature_exposure_fixed": bool(feature_audit.get("candidate_feature_signal_missing") is False),
        "behavior_logprob_contract_fixed": bool(behavior_audit.get("behavior_logprob_recomputable") is True),
        "synthetic_credit_target_selected_count": int(
            sum(1 for row in target_rows if row.get("synthetic_credit_target_selected") is True)
        ),
        "min_synthetic_credit_target_selected_count": int(config["min_synthetic_credit_target_selected_count"]),
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
        "action_space_type": ACTION_SPACE_TYPE,
        "max_traversable_slope_deg": 30.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str(output_root / SUMMARY_FILE),
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "summary_status": status,
        "next_required_change": route,
        "artifacts": {
            "summary": str(output_root / SUMMARY_FILE),
            "synthetic_credit_target_audit": str(output_root / TARGET_AUDIT_FILE),
            "behavior_logprob_audit": str(output_root / BEHAVIOR_AUDIT_FILE),
            "candidate_feature_exposure_audit": str(output_root / FEATURE_AUDIT_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
    }
    _write_jsonl(output_root / TARGET_AUDIT_FILE, target_rows)
    _write_json(output_root / BEHAVIOR_AUDIT_FILE, behavior_audit)
    _write_json(output_root / FEATURE_AUDIT_FILE, feature_audit)
    _write_json(output_root / STAGE26_1_SUMMARY_FILE, stage26_1_summary)
    _write_json(output_root / STAGE26_2_SUMMARY_FILE, stage26_2_summary)
    _write_json(output_root / STAGE26_3_SUMMARY_FILE, stage26_3_summary)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    config["stage26_5_root"] = str(_resolve_path(Path(config.get("stage26_5_root") or DEFAULT_STAGE26_5_ROOT), repo_root))
    config["stage26_0_root"] = str(_resolve_path(Path(config.get("stage26_0_root") or DEFAULT_STAGE26_0_ROOT), repo_root))
    config["stage26_1_base_config"] = str(
        _resolve_path(Path(config.get("stage26_1_base_config") or stage26_1.DEFAULT_CONFIG), repo_root)
    )
    config["stage26_2_base_config"] = str(
        _resolve_path(Path(config.get("stage26_2_base_config") or stage26_2.DEFAULT_CONFIG), repo_root)
    )
    config["stage26_3_base_config"] = str(
        _resolve_path(Path(config.get("stage26_3_base_config") or stage26_3.DEFAULT_CONFIG), repo_root)
    )
    config["run_stage26_chain"] = bool(config.get("run_stage26_chain", True))
    config["synthetic_credit_mixture_probability"] = _fraction(
        config.get("synthetic_credit_mixture_probability", 0.35),
        "synthetic_credit_mixture_probability",
    )
    config["min_synthetic_credit_target_selected_count"] = int(
        config.get("min_synthetic_credit_target_selected_count", 4)
    )
    config["hybrid_astar_candidate_eval_workers"] = int(config.get("hybrid_astar_candidate_eval_workers", 4))
    config["trainable_transition_target_count"] = int(config.get("trainable_transition_target_count", 16))
    config["required_scenario_count"] = int(config.get("required_scenario_count", 2))
    config["rollout_steps"] = int(config.get("rollout_steps", 8))
    config["eval_rollout_steps"] = int(config.get("eval_rollout_steps", 4))
    config["epochs"] = int(config.get("epochs", 8))
    config["learning_rate"] = float(config.get("learning_rate", 2.0e-5))
    config["policy_loss_coefficient"] = float(config.get("policy_loss_coefficient", 2.0))
    config["value_loss_coefficient"] = float(config.get("value_loss_coefficient", 0.02))
    config["loss_scale"] = float(config.get("loss_scale", 0.5))
    config["clip_ratio"] = float(config.get("clip_ratio", 0.2))
    config["entropy_coefficient"] = float(config.get("entropy_coefficient", 0.01))
    config["advantage_clip_abs"] = float(config.get("advantage_clip_abs", 5.0))
    config["max_grad_norm"] = float(config.get("max_grad_norm", 1.0))
    config["max_abs_approx_kl"] = float(config.get("max_abs_approx_kl", 0.5))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    return config


def _run_stage26_chain(*, config: dict[str, Any], output_root: Path, repo_root: Path) -> dict[str, Any]:
    stage26_1_config = _build_stage26_1_config(config)
    stage26_1_config_path = output_root / "xunce-stage26-6-stage26-1-config.json"
    _write_json(stage26_1_config_path, stage26_1_config)
    stage26_1_summary = stage26_1.run_xunce_stage26_1_synthetic_terrain_collector_smoke(
        config_path=stage26_1_config_path,
        output_root=output_root / "s26_1",
        repo_root=repo_root,
    )
    target_rows = _target_rows_from_stage26_1(output_root / "s26_1")
    feature_audit = _feature_audit_from_stage26_1(output_root / "s26_1", target_rows)
    behavior_audit = _behavior_audit_from_stage26_1(output_root / "s26_1", target_rows)
    stage26_2_summary: dict[str, Any] = {}
    stage26_3_summary: dict[str, Any] = {}
    if stage26_1_summary.get("status") == "passed":
        stage26_2_config = _build_stage26_2_config(config, output_root / "s26_1")
        stage26_2_config_path = output_root / "xunce-stage26-6-stage26-2-config.json"
        _write_json(stage26_2_config_path, stage26_2_config)
        stage26_2_summary = stage26_2.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
            config_path=stage26_2_config_path,
            output_root=output_root / "s26_2",
            repo_root=repo_root,
        )
    if stage26_2_summary.get("status") == "passed":
        stage26_3_config = _build_stage26_3_config(config, output_root / "s26_2")
        stage26_3_config_path = output_root / "xunce-stage26-6-stage26-3-config.json"
        _write_json(stage26_3_config_path, stage26_3_config)
        stage26_3_summary = stage26_3.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
            config_path=stage26_3_config_path,
            output_root=output_root / "s26_3",
            repo_root=repo_root,
        )
    return {
        "stage26_1_summary": stage26_1_summary,
        "stage26_2_summary": stage26_2_summary,
        "stage26_3_summary": stage26_3_summary,
        "target_rows": target_rows,
        "feature_audit": feature_audit,
        "behavior_audit": behavior_audit,
    }


def _build_stage26_1_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = _read_json(Path(config["stage26_1_base_config"]))
    cfg.update(
        {
            "stage26_0_root": config["stage26_0_root"],
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "min_trainable_transition_count": int(config["trainable_transition_target_count"]),
            "action_space_type": ACTION_SPACE_TYPE,
            "continuous_theta_action_space_enabled": True,
            "synthetic_credit_feature_exposure_enabled": True,
            "synthetic_exploration_credit_enabled": True,
            "synthetic_credit_mixture_probability": float(config["synthetic_credit_mixture_probability"]),
            "allow_synthetic_credit_behavior_policy": True,
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _build_stage26_2_config(config: dict[str, Any], stage26_1_root: Path) -> dict[str, Any]:
    cfg = _read_json(Path(config["stage26_2_base_config"]))
    cfg.update(
        {
            "stage26_1_root": str(stage26_1_root),
            "epochs": int(config["epochs"]),
            "learning_rate": float(config["learning_rate"]),
            "clip_ratio": float(config["clip_ratio"]),
            "policy_loss_coefficient": float(config["policy_loss_coefficient"]),
            "value_loss_coefficient": float(config["value_loss_coefficient"]),
            "entropy_coefficient": float(config["entropy_coefficient"]),
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
            "normalize_minibatch_advantages": True,
            "loss_scale": float(config["loss_scale"]),
            "max_grad_norm": float(config["max_grad_norm"]),
            "max_abs_approx_kl": float(config["max_abs_approx_kl"]),
            "stage26_2_authorized": False,
            "runs_new_ppo_update": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _build_stage26_3_config(config: dict[str, Any], stage26_2_root: Path) -> dict[str, Any]:
    cfg = _read_json(Path(config["stage26_3_base_config"]))
    cfg.update(
        {
            "stage26_2_root": str(stage26_2_root),
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["eval_rollout_steps"]),
            "action_space_type": ACTION_SPACE_TYPE,
            "continuous_theta_action_space_enabled": True,
            "synthetic_credit_feature_exposure_enabled": True,
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
            "stage26_3_authorized": False,
            "release_or_training_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    config: dict[str, Any],
    feature_audit: dict[str, Any],
    behavior_audit: dict[str, Any],
    target_rows: list[dict[str, Any]],
    stage26_1_summary: dict[str, Any],
    stage26_2_summary: dict[str, Any],
    stage26_3_summary: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if feature_audit.get("candidate_feature_signal_missing") is True:
        return ROUTE_FEATURE
    target_selected = sum(1 for row in target_rows if row.get("synthetic_credit_target_selected") is True)
    if target_selected < int(config["min_synthetic_credit_target_selected_count"]):
        return ROUTE_SAMPLER
    if behavior_audit.get("behavior_logprob_recomputable") is not True:
        return ROUTE_LOGPROB
    if stage26_2_summary and stage26_2_summary.get("status") != "passed":
        return ROUTE_STABILITY
    if stage26_3_summary and stage26_3_summary.get("status") != "passed":
        if (
            stage26_3_summary.get("next_required_change") == "rerun_stage26_3_required_inputs"
            or stage26_3_summary.get("stage21_5_execution_incomplete") is True
            or int(stage26_3_summary.get("synthetic_inference_required_field_missing_count", 0) or 0) > 0
            or int(stage26_3_summary.get("hybrid_path_contract_mismatch_count", 0) or 0) > 0
        ):
            return ROUTE_FEATURE
    stage26_3_route = str(stage26_3_summary.get("next_required_change") or "")
    if stage26_3_route in {ROUTE_STAGE26_7, ROUTE_STAGE26_3_MULTI_SEED_ALIAS}:
        return ROUTE_STAGE26_7
    if stage26_3_route.startswith("repair_stage26_3_"):
        return stage26_3_route
    if stage26_3_route == "repair_stage26_synthetic_policy_update_signal_strength":
        return ROUTE_GRADIENT
    if stage26_3_route == "calibrate_stage26_synthetic_discrete_margin_crossing":
        return ROUTE_MARGIN
    if stage26_3_route == ROUTE_CREDIT:
        return ROUTE_CREDIT
    return ROUTE_GRADIENT


def _target_rows_from_stage26_1(stage26_1_root: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl_if_exists(stage26_1_root / "s21_1" / "xunce-stage21-1-ppo-trainable-batch.jsonl")
    target_rows: list[dict[str, Any]] = []
    for row in rows:
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        semantic_map = info.get("xunce_batch_feature_semantic_map")
        feature_names = semantic_map.get("feature_names") if isinstance(semantic_map, dict) else []
        feature_nonzero_counts = _candidate_feature_nonzero_counts(row, feature_names)
        target_rows.append(
            {
                "schema_version": "xunce-stage26-6-synthetic-credit-target-audit-row/v1",
                "transition_id": row.get("transition_id"),
                "scenario_id": row.get("scenario_id"),
                "step_index": row.get("step_index"),
                "action_index": row.get("action_index"),
                "synthetic_credit_target_index": row.get(
                    "synthetic_credit_target_index",
                    info.get("synthetic_credit_target_index"),
                ),
                "synthetic_credit_target_selected": row.get(
                    "synthetic_credit_target_selected",
                    info.get("synthetic_credit_target_selected"),
                ),
                "synthetic_credit_score": row.get("synthetic_credit_score", info.get("synthetic_credit_score")),
                "behavior_policy_id": row.get("behavior_policy_id", info.get("behavior_policy_id")),
                "old_log_prob": row.get("old_log_prob"),
                "old_behavior_log_prob": row.get("old_behavior_log_prob", info.get("old_behavior_log_prob")),
                "xunce_batch_feature_semantic_map": semantic_map,
                "candidate_feature_nonzero_counts": feature_nonzero_counts,
            }
        )
    return target_rows


def _feature_audit_from_stage26_1(stage26_1_root: Path, target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing = 0
    coverage_signal_rows = 0
    hybrid_signal_rows = 0
    synthetic_signal_rows = 0
    for row in target_rows:
        semantic_map = row.get("xunce_batch_feature_semantic_map")
        if not (
            isinstance(semantic_map, dict)
            and semantic_map.get("feature_contract_id") == "synthetic_credit_candidate_features/v1"
            and isinstance(semantic_map.get("feature_names"), list)
            and "hybrid_astar_path_cost_norm" in semantic_map.get("feature_names")
            and "synthetic_los_blocker_pressure_norm" in semantic_map.get("feature_names")
            and "synthetic_hard_obstacle_pressure_norm" in semantic_map.get("feature_names")
        ):
            missing += 1
            continue
        nonzero = row.get("candidate_feature_nonzero_counts")
        if not isinstance(nonzero, dict):
            missing += 1
            continue
        if (
            int(nonzero.get("obstacle_aware_new_visible_norm", 0) or 0) > 0
            or int(nonzero.get("obstacle_aware_gain_per_hybrid_cost_norm", 0) or 0) > 0
        ):
            coverage_signal_rows += 1
        if int(nonzero.get("hybrid_astar_path_cost_norm", 0) or 0) > 0:
            hybrid_signal_rows += 1
        if (
            int(nonzero.get("synthetic_los_blocker_pressure_norm", 0) or 0) > 0
            or int(nonzero.get("synthetic_hard_obstacle_pressure_norm", 0) or 0) > 0
        ):
            synthetic_signal_rows += 1
    return {
        "schema_version": "xunce-stage26-6-candidate-feature-exposure-audit/v1",
        "stage26_1_root": str(stage26_1_root),
        "row_count": len(target_rows),
        "feature_semantic_map_missing_count": missing,
        "coverage_signal_row_count": coverage_signal_rows,
        "hybrid_path_signal_row_count": hybrid_signal_rows,
        "synthetic_pressure_signal_row_count": synthetic_signal_rows,
        "candidate_feature_signal_missing": (
            missing > 0
            or not target_rows
            or coverage_signal_rows == 0
            or hybrid_signal_rows == 0
            or synthetic_signal_rows == 0
        ),
        "network_structure_changed": False,
    }


def _candidate_feature_nonzero_counts(row: dict[str, Any], feature_names: Any) -> dict[str, int]:
    if not isinstance(feature_names, list):
        return {}
    batch = row.get("xunce_batch") if isinstance(row.get("xunce_batch"), dict) else {}
    candidate_features = batch.get("candidate_features") if isinstance(batch.get("candidate_features"), dict) else {}
    values = candidate_features.get("values")
    if not (isinstance(values, list) and values and isinstance(values[0], list)):
        return {}
    rows = values[0]
    result: dict[str, int] = {}
    for column, name in enumerate(feature_names):
        count = 0
        for feature_row in rows:
            if not isinstance(feature_row, list) or column >= len(feature_row):
                continue
            value = _finite(feature_row[column])
            if value is not None and abs(value) > 1.0e-9:
                count += 1
        result[str(name)] = count
    return result


def _behavior_audit_from_stage26_1(stage26_1_root: Path, target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage21_3_summary = _read_json_if_exists(stage26_1_root / "s21_3" / "xunce-stage21-3-ppo-batch-validation-summary.json")
    behavior_rows = [row for row in target_rows if row.get("behavior_policy_id") == "synthetic_credit_mixture_policy/v1"]
    mismatch = 0
    for row in behavior_rows:
        old_total = _finite(row.get("old_log_prob"))
        old_behavior = _finite(row.get("old_behavior_log_prob"))
        if old_total is None or old_behavior is None or abs(old_total - old_behavior) > 1.0e-5:
            mismatch += 1
    missing_count = int(stage21_3_summary.get("synthetic_credit_behavior_logprob_missing_count", mismatch) or 0)
    return {
        "schema_version": "xunce-stage26-6-behavior-logprob-audit/v1",
        "stage26_1_root": str(stage26_1_root),
        "behavior_policy_id": "synthetic_credit_mixture_policy/v1",
        "behavior_policy_row_count": len(behavior_rows),
        "behavior_logprob_mismatch_count": mismatch,
        "stage21_3_behavior_logprob_missing_count": missing_count,
        "behavior_logprob_recomputable": len(behavior_rows) > 0 and mismatch == 0 and missing_count == 0,
    }


def _input_rejections(stage26_5_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not stage26_5_summary:
        reasons.append("missing_stage26_5_summary")
        return reasons
    if stage26_5_summary.get("status") != "failed":
        reasons.append("stage26_5_status_not_failed")
    if stage26_5_summary.get("next_required_change") != ROUTE_FROM_STAGE26_5:
        reasons.append("stage26_5_route_mismatch")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is not False]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _read_stage26_5_summary(root: Path) -> dict[str, Any]:
    path = root / "xunce-stage26-5-summary.json"
    return _read_json(path) if path.is_file() else {}


def _empty_feature_audit() -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage26-6-candidate-feature-exposure-audit/v1",
        "candidate_feature_signal_missing": False,
        "network_structure_changed": False,
    }


def _empty_behavior_audit() -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage26-6-behavior-logprob-audit/v1",
        "behavior_logprob_recomputable": True,
        "behavior_policy_id": "synthetic_credit_mixture_policy/v1",
    }


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.6 Synthetic Exploration Credit Assignment",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- synthetic_credit_target_selected_count: `{summary['synthetic_credit_target_selected_count']}`",
            f"- feature_exposure_fixed: `{summary['feature_exposure_fixed']}`",
            f"- behavior_logprob_contract_fixed: `{summary['behavior_logprob_contract_fixed']}`",
            "",
            "本阶段只证明 synthetic credit 的合同和 smoke 链路，不发布 checkpoint、不替换 default policy。",
        ]
    )


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _fraction(value: Any, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
