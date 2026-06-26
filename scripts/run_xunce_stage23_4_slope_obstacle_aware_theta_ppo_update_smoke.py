from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage21_4_tiny_ppo_update_smoke as stage21_4
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage21_4_tiny_ppo_update_smoke as stage21_4


CONFIG_SCHEMA_VERSION = "xunce-stage23-4-slope-obstacle-aware-theta-ppo-update-smoke-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage23-4-summary/v1"
BATCH_AUDIT_SCHEMA_VERSION = "xunce-stage23-4-slope-theta-ppo-batch-update-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage23-4-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage23-4-manifest/v1"

STAGE_ID = "xunce-stage23-4-slope-obstacle-aware-theta-ppo-update-smoke"
DEFAULT_CONFIG = "configs/xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage23_slope_obstacle_aware_theta_reward/"
    "outputs/path_feedback_batch_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke_v1"
)
DEFAULT_STAGE23_3_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage23_slope_obstacle_aware_theta_reward/"
    "outputs/path_feedback_batch_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke_v1"
)

SUMMARY_FILE = "xunce-stage23-4-summary.json"
STAGE21_4_CONFIG_FILE = "xunce-stage23-4-stage21-4-config.json"
STAGE21_4_SUMMARY_FILE = "xunce-stage23-4-stage21-4-summary.json"
BATCH_AUDIT_FILE = "xunce-stage23-4-slope-theta-ppo-batch-update-audit.json"
LOSS_GRADIENT_AUDIT_FILE = "xunce-stage23-4-loss-gradient-audit.json"
CHECKPOINT_BOUNDARY_AUDIT_FILE = "xunce-stage23-4-checkpoint-boundary-audit.json"
ROUTING_FILE = "xunce-stage23-4-next-stage-routing.json"
REPORT_FILE = "xunce-stage23-4-report.md"
MANIFEST_FILE = "xunce-stage23-4-manifest.json"

SLOPE_COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
ROUTE_INPUTS = "rerun_stage23_4_required_inputs"
ROUTE_BATCH = "repair_stage23_4_slope_theta_batch_update_contract"
ROUTE_STABILITY = "repair_stage23_4_slope_theta_ppo_update_stability"
ROUTE_CHECKPOINT = "repair_stage23_4_checkpoint_reload_boundary"
ROUTE_STAGE23_5 = "run_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke"
ROUTE_BOUNDARY = "resolve_stage23_4_boundary_rejections"
ROUTE_FROM_STAGE23_3 = "run_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke"
ROUTE_FROM_STAGE21_4 = "implement_stage21_5_single_seed_ppo_pilot"

BOUNDARY_FALSE_FIELDS = (
    "stage23_4_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage23.4 slope-obstacle-aware theta PPO update smoke.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    boundary_reasons = _boundary_rejections(config)
    input_reasons = _input_rejections(config)
    batch_rows = _read_jsonl_if_exists(Path(config["stage23_3_root"]) / "s21_3" / "xunce-stage21-3-ppo-trainable-batch.jsonl")
    batch_audit = _slope_theta_batch_update_audit(batch_rows)

    stage21_4_summary: dict[str, Any] = {}
    stage21_4_config_path = output_root / STAGE21_4_CONFIG_FILE
    if not boundary_reasons and not input_reasons and _batch_contract_passed(batch_audit):
        stage21_4_summary = _run_stage21_4(config, output_root, repo_root, stage21_4_config_path)

    stage21_4_root = output_root / "s21_4"
    loss_rows = _read_jsonl_if_exists(stage21_4_root / stage21_4.LOSS_AUDIT_FILE)
    gradient_audit = _read_json_if_exists(stage21_4_root / stage21_4.GRADIENT_AUDIT_FILE)
    checkpoint_audit = _read_json_if_exists(stage21_4_root / stage21_4.CHECKPOINT_AUDIT_FILE)
    loss_gradient_audit = _loss_gradient_audit(loss_rows, gradient_audit, stage21_4_summary)
    checkpoint_boundary_audit = _checkpoint_boundary_audit(
        checkpoint_audit,
        expected_stage21_3_root=Path(config["stage23_3_root"]) / "s21_3",
        expected_stage21_4_root=stage21_4_root,
        stage21_4_summary=stage21_4_summary,
    )

    status, route, route_reason = _route(
        boundary_reasons,
        input_reasons,
        batch_audit,
        stage21_4_summary,
        loss_gradient_audit,
        checkpoint_boundary_audit,
    )
    blocking = _unique(
        boundary_reasons
        + input_reasons
        + _route_blocking_reasons(route, batch_audit, stage21_4_summary, loss_gradient_audit, checkpoint_boundary_audit)
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking,
        "stage23_3_root": config["stage23_3_root"],
        "stage21_4_root": str(stage21_4_root),
        "stage21_4_status": stage21_4_summary.get("status"),
        "stage21_4_next_required_change": stage21_4_summary.get("next_required_change"),
        "coverage_source": SLOPE_COVERAGE_SOURCE,
        "max_traversable_slope_deg": 30.0,
        **_summary_counts(batch_audit, loss_gradient_audit, checkpoint_boundary_audit),
        "stage23_4_authorized": False,
        "runs_new_ppo_update": bool(status == "passed"),
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
        "stage23_4_authorized": False,
        "runs_new_ppo_update": bool(status == "passed"),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "summary": str(output_root / SUMMARY_FILE),
        "stage21_4_config": str(stage21_4_config_path),
        "stage21_4_summary": str(output_root / STAGE21_4_SUMMARY_FILE),
        "slope_theta_ppo_batch_update_audit": str(output_root / BATCH_AUDIT_FILE),
        "loss_gradient_audit": str(output_root / LOSS_GRADIENT_AUDIT_FILE),
        "checkpoint_boundary_audit": str(output_root / CHECKPOINT_BOUNDARY_AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / STAGE21_4_SUMMARY_FILE, stage21_4_summary)
    _write_json(output_root / BATCH_AUDIT_FILE, batch_audit)
    _write_json(output_root / LOSS_GRADIENT_AUDIT_FILE, loss_gradient_audit)
    _write_json(output_root / CHECKPOINT_BOUNDARY_AUDIT_FILE, checkpoint_boundary_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _run_stage21_4(config: dict[str, Any], output_root: Path, repo_root: Path, config_path: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage21_4_base_config"], repo_root)
    cfg.update(
        {
            "stage21_3_ppo_batch_validation_root": str(Path(config["stage23_3_root"]) / "s21_3"),
            "xunce_candidate_checkpoint": config["xunce_candidate_checkpoint"],
            "high_fidelity_config": config["high_fidelity_config"],
            "epochs": int(config["epochs"]),
            "learning_rate": float(config["learning_rate"]),
            "clip_ratio": float(config["clip_ratio"]),
            "policy_loss_coefficient": float(config["policy_loss_coefficient"]),
            "value_loss_coefficient": float(config["value_loss_coefficient"]),
            "entropy_coefficient": float(config["entropy_coefficient"]),
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
            "normalize_minibatch_advantages": bool(config["normalize_minibatch_advantages"]),
            "loss_scale": float(config["loss_scale"]),
            "max_grad_norm": float(config["max_grad_norm"]),
            "max_abs_approx_kl": float(config["max_abs_approx_kl"]),
            "offline_ppo_update_smoke_authorized": True,
            "stage21_4_authorized": False,
            "runs_new_ppo_update": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    _write_json(config_path, cfg)
    return stage21_4.run_xunce_stage21_4_tiny_ppo_update_smoke(
        config_path=config_path,
        output_root=output_root / "s21_4",
        repo_root=repo_root,
    )


def _slope_theta_batch_update_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing = 0
    shape_mismatch = 0
    action_binding_mismatch = 0
    selected_mask_violation = 0
    fallback = 0
    selected_info_mismatch = 0
    slope_array_length_mismatch = 0
    slope_binding_mismatch = 0
    for row in rows:
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        viewpoints = info.get("candidate_viewpoints")
        action_index = _int_or_none(row.get("action_index"))
        if not _row_has_slope_contract(row, info):
            missing += 1
            continue
        if row.get("point_only_reward_fallback_used") is True or row.get("unobstructed_theta_reward_fallback_used") is True:
            fallback += 1
        if action_index is None or action_index < 0 or action_index >= len(viewpoints):
            action_binding_mismatch += 1
            continue
        if list(viewpoints[action_index]) != list(row.get("candidate_viewpoint") or []):
            action_binding_mismatch += 1
        if _int_or_none(row.get("candidate_theta_deg")) != _theta_from_viewpoint(row.get("candidate_viewpoint")):
            action_binding_mismatch += 1
        if not _selected_info_matches(row, info, action_index):
            selected_info_mismatch += 1
        action_count = len(viewpoints)
        length_ok, binding_ok = _slope_provenance_arrays_match(row, info, action_index, action_count)
        if not length_ok:
            slope_array_length_mismatch += 1
        if not binding_ok:
            slope_binding_mismatch += 1
        xunce_batch = row.get("xunce_batch") if isinstance(row.get("xunce_batch"), dict) else {}
        if not (
            _tensor_second_dim_is(xunce_batch.get("action_mask"), action_count)
            and _tensor_second_dim_is(xunce_batch.get("candidate_features"), action_count)
            and _tensor_second_dim_is(xunce_batch.get("context_features"), action_count)
            and _tensor_second_dim_is(xunce_batch.get("candidate_missing_indicators"), action_count)
        ):
            shape_mismatch += 1
        for mask_name in ("action_mask", "sampling_mask", "hard_risk_clean_mask"):
            if _mask_allows(info, mask_name, action_index) is not True:
                selected_mask_violation += 1
    return {
        "schema_version": BATCH_AUDIT_SCHEMA_VERSION,
        "batch_row_count": len(rows),
        "slope_theta_batch_contract_missing_count": missing,
        "xunce_batch_viewpoint_shape_mismatch_count": shape_mismatch,
        "action_viewpoint_binding_mismatch_count": action_binding_mismatch,
        "selected_info_viewpoint_mismatch_count": selected_info_mismatch,
        "slope_provenance_array_length_mismatch_count": slope_array_length_mismatch,
        "slope_provenance_binding_mismatch_count": slope_binding_mismatch,
        "selected_action_mask_violation_count": selected_mask_violation,
        "reward_fallback_used_count": fallback,
    }


def _row_has_slope_contract(row: dict[str, Any], info: dict[str, Any]) -> bool:
    return (
        row.get("theta_aware_reward_contract") is True
        and row.get("slope_obstacle_aware_theta_reward_contract") is True
        and row.get("coverage_source") == SLOPE_COVERAGE_SOURCE
        and row.get("point_only_reward_fallback_used") is False
        and row.get("unobstructed_theta_reward_fallback_used") is False
        and isinstance(row.get("candidate_viewpoint"), list)
        and row.get("candidate_theta_deg") is not None
        and _finite_float(row.get("obstacle_aware_new_visible_cell_count")) is not None
        and bool(str(row.get("obstacle_aware_theta_coverage_hash") or "").strip())
        and _finite_float(row.get("obstacle_aware_theta_coverage_gain_per_path_cost")) is not None
        and _finite_float(row.get("obstacle_aware_theta_coverage_denominator_cells")) is not None
        and bool(str(row.get("slope_obstacle_source_hash") or "").strip())
        and bool(str(row.get("platform_contract_hash") or "").strip())
        and _finite_float(row.get("max_traversable_slope_deg")) == 30.0
        and row.get("slope_blocked_source_kind") == "slope_blocked_as_obstacle_proxy"
        and isinstance(info.get("candidate_viewpoints"), list)
        and isinstance(info.get("candidate_theta_deg"), list)
        and isinstance(info.get("obstacle_aware_new_visible_cell_counts"), list)
        and isinstance(info.get("obstacle_aware_theta_coverage_hashes"), list)
        and isinstance(info.get("obstacle_aware_theta_coverage_gain_per_path_costs"), list)
        and _slope_reward_metrics_consistent(row)
    )


def _slope_reward_metrics_consistent(row: dict[str, Any]) -> bool:
    metrics = row.get("reward_metrics") if isinstance(row.get("reward_metrics"), dict) else row.get("metrics")
    if not isinstance(metrics, dict):
        return False
    coverage_rate_delta = _finite_float(metrics.get("coverage_rate_delta"))
    denominator = _finite_float(row.get("obstacle_aware_theta_coverage_denominator_cells"))
    obstacle_new = _finite_float(row.get("obstacle_aware_new_visible_cell_count"))
    if coverage_rate_delta is None or denominator is None or denominator <= 0.0 or obstacle_new is None:
        return False
    return abs(coverage_rate_delta - (obstacle_new / denominator)) <= 1.0e-9


def _loss_gradient_audit(
    loss_rows: list[dict[str, Any]],
    gradient: dict[str, Any],
    stage21_4_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    finite_loss = bool(loss_rows) and all(
        _finite_float(row.get(key)) is not None
        for row in loss_rows
        for key in ("total_loss", "policy_loss", "value_loss", "entropy", "post_update_approx_kl", "post_update_clip_fraction")
    )
    summary = stage21_4_summary or {}
    parameter_delta_l2 = _finite_float(gradient.get("parameter_delta_l2"))
    if parameter_delta_l2 is None:
        parameter_delta_l2 = _finite_float(summary.get("parameter_delta_l2"))
    finite_gradient = (
        _finite_float(gradient.get("pre_clip_grad_norm")) is not None
        and _finite_float(gradient.get("post_clip_grad_norm")) is not None
        and parameter_delta_l2 is not None
        and parameter_delta_l2 > 0.0
    )
    component_norms = gradient.get("component_grad_norms") if isinstance(gradient.get("component_grad_norms"), dict) else {}
    component_grad_norms_readable = all(
        _finite_float(component_norms.get(key)) is not None
        for key in ("total_loss_grad_norm", "policy_loss_grad_norm", "value_loss_grad_norm", "entropy_loss_grad_norm")
    )
    return {
        "loss_row_count": len(loss_rows),
        "loss_finite": finite_loss,
        "gradient_finite": finite_gradient,
        "component_grad_norms_readable": component_grad_norms_readable,
        "final_total_loss": _finite_float(loss_rows[-1].get("total_loss")) if loss_rows else None,
        "final_post_update_approx_kl": _finite_float(loss_rows[-1].get("post_update_approx_kl")) if loss_rows else None,
        "pre_clip_grad_norm": _finite_float(gradient.get("pre_clip_grad_norm")),
        "post_clip_grad_norm": _finite_float(gradient.get("post_clip_grad_norm")),
        "parameter_delta_l2": parameter_delta_l2,
        "parameter_with_grad_count": _int_or_none(gradient.get("parameter_with_grad_count")),
        "component_grad_norms": component_norms,
    }


def _selected_info_matches(row: dict[str, Any], info: dict[str, Any], action_index: int) -> bool:
    return (
        isinstance(info.get("selected_viewpoint"), list)
        and list(info.get("selected_viewpoint")) == list(row.get("candidate_viewpoint") or [])
        and _int_or_none(info.get("selected_theta_deg")) == _int_or_none(row.get("candidate_theta_deg"))
        and isinstance(info.get("candidate_theta_deg"), list)
        and action_index < len(info["candidate_theta_deg"])
        and _int_or_none(info["candidate_theta_deg"][action_index]) == _int_or_none(row.get("candidate_theta_deg"))
    )


def _slope_provenance_arrays_match(row: dict[str, Any], info: dict[str, Any], action_index: int, action_count: int) -> tuple[bool, bool]:
    checks = (
        ("obstacle_aware_new_visible_cell_counts", row.get("obstacle_aware_new_visible_cell_count")),
        ("obstacle_aware_theta_coverage_hashes", row.get("obstacle_aware_theta_coverage_hash")),
        ("obstacle_aware_theta_coverage_gain_per_path_costs", row.get("obstacle_aware_theta_coverage_gain_per_path_cost")),
        ("slope_obstacle_source_hashes", row.get("slope_obstacle_source_hash")),
        ("platform_contract_hashes", row.get("platform_contract_hash")),
    )
    length_ok = True
    binding_ok = True
    for key, expected in checks:
        values = info.get(key)
        if not isinstance(values, list) or len(values) != action_count:
            length_ok = False
            binding_ok = False
            continue
        actual = values[action_index]
        if _finite_float(expected) is not None or _finite_float(actual) is not None:
            if _finite_float(actual) != _finite_float(expected):
                binding_ok = False
        elif actual != expected:
            binding_ok = False
    if _finite_float(info.get("max_traversable_slope_deg")) != 30.0:
        binding_ok = False
    if info.get("slope_blocked_source_kind") != "slope_blocked_as_obstacle_proxy":
        binding_ok = False
    return length_ok, binding_ok


def _checkpoint_boundary_audit(
    checkpoint: dict[str, Any],
    *,
    expected_stage21_3_root: Path,
    expected_stage21_4_root: Path,
    stage21_4_summary: dict[str, Any],
) -> dict[str, Any]:
    metadata = checkpoint.get("metadata") if isinstance(checkpoint.get("metadata"), dict) else {}
    boundary_false = all(
        metadata.get(key) is False
        for key in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary", "training_or_release_authorized")
    )
    canary_zero = float(metadata.get("canary_traffic_fraction", 0.0) or 0.0) == 0.0
    lineage_stage21_3 = _same_resolved_path(metadata.get("stage21_3_ppo_batch_validation_root"), expected_stage21_3_root)
    checkpoint_in_root = _path_parent_is(checkpoint.get("experimental_checkpoint_path"), expected_stage21_4_root)
    metadata_source_sha = metadata.get("source_checkpoint_sha256")
    summary_source_sha = stage21_4_summary.get("source_xunce_checkpoint_sha256")
    source_sha_consistent = bool(metadata_source_sha and summary_source_sha and metadata_source_sha == summary_source_sha)
    return {
        "checkpoint_exists": checkpoint.get("experimental_checkpoint_exists") is True,
        "checkpoint_reload_passed": checkpoint.get("checkpoint_reload_passed") is True,
        "experimental_only": metadata.get("experimental_only") is True,
        "experimental_checkpoint_sha256": checkpoint.get("experimental_checkpoint_sha256"),
        "stage21_3_lineage_passed": lineage_stage21_3,
        "checkpoint_path_in_stage21_4_root": checkpoint_in_root,
        "source_checkpoint_sha_consistent": source_sha_consistent,
        "checkpoint_boundary_passed": bool(
            checkpoint.get("experimental_checkpoint_exists") is True
            and checkpoint.get("checkpoint_reload_passed") is True
            and metadata.get("experimental_only") is True
            and lineage_stage21_3
            and checkpoint_in_root
            and source_sha_consistent
            and boundary_false
            and canary_zero
        ),
        "metadata": metadata,
    }


def _route(
    boundary_reasons: list[str],
    input_reasons: list[str],
    batch_audit: dict[str, Any],
    stage21_4_summary: dict[str, Any],
    loss_gradient_audit: dict[str, Any],
    checkpoint_audit: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage23_4_boundary_rejected"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage23_4_inputs_missing_or_untrusted"
    if not _batch_contract_passed(batch_audit):
        return "failed", ROUTE_BATCH, "slope_theta_batch_update_contract_failed"
    if (
        stage21_4_summary.get("status") != "passed"
        or stage21_4_summary.get("next_required_change") != ROUTE_FROM_STAGE21_4
        or not loss_gradient_audit["loss_finite"]
        or not loss_gradient_audit["gradient_finite"]
        or not loss_gradient_audit["component_grad_norms_readable"]
    ):
        return "failed", ROUTE_STABILITY, "slope_theta_ppo_update_stability_failed"
    if not checkpoint_audit["checkpoint_boundary_passed"]:
        return "failed", ROUTE_CHECKPOINT, "slope_theta_checkpoint_reload_boundary_failed"
    return "passed", ROUTE_STAGE23_5, "slope_obstacle_aware_theta_ppo_update_smoke_ready"


def _batch_contract_passed(audit: dict[str, Any]) -> bool:
    return (
        audit["batch_row_count"] > 0
        and audit["slope_theta_batch_contract_missing_count"] == 0
        and audit["xunce_batch_viewpoint_shape_mismatch_count"] == 0
        and audit["action_viewpoint_binding_mismatch_count"] == 0
        and audit["selected_info_viewpoint_mismatch_count"] == 0
        and audit["slope_provenance_array_length_mismatch_count"] == 0
        and audit["slope_provenance_binding_mismatch_count"] == 0
        and audit["selected_action_mask_violation_count"] == 0
        and audit["reward_fallback_used_count"] == 0
    )


def _route_blocking_reasons(
    route: str,
    batch_audit: dict[str, Any],
    stage21_4_summary: dict[str, Any],
    loss_gradient_audit: dict[str, Any],
    checkpoint_audit: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if route == ROUTE_BATCH:
        if batch_audit["batch_row_count"] <= 0:
            reasons.append("slope_theta_ppo_batch_rows_missing")
        for key in (
            "slope_theta_batch_contract_missing_count",
            "xunce_batch_viewpoint_shape_mismatch_count",
            "action_viewpoint_binding_mismatch_count",
            "selected_info_viewpoint_mismatch_count",
            "slope_provenance_array_length_mismatch_count",
            "slope_provenance_binding_mismatch_count",
            "selected_action_mask_violation_count",
            "reward_fallback_used_count",
        ):
            if batch_audit[key]:
                reasons.append(key.removesuffix("_count"))
    if route == ROUTE_STABILITY:
        if stage21_4_summary.get("status") != "passed":
            reasons.append("stage21_4_not_passed")
        if stage21_4_summary.get("next_required_change") != ROUTE_FROM_STAGE21_4:
            reasons.append("stage21_4_route_not_stage21_5")
        for key in ("loss_finite", "gradient_finite", "component_grad_norms_readable"):
            if not loss_gradient_audit[key]:
                reasons.append(key)
    if route == ROUTE_CHECKPOINT:
        if not checkpoint_audit["checkpoint_exists"]:
            reasons.append("experimental_checkpoint_missing")
        if not checkpoint_audit["checkpoint_reload_passed"]:
            reasons.append("checkpoint_reload_failed")
        if not checkpoint_audit["experimental_only"]:
            reasons.append("checkpoint_not_experimental_only")
        if not checkpoint_audit["stage21_3_lineage_passed"]:
            reasons.append("checkpoint_stage21_3_lineage_mismatch")
        if not checkpoint_audit["checkpoint_path_in_stage21_4_root"]:
            reasons.append("checkpoint_path_outside_stage21_4_root")
        if not checkpoint_audit["source_checkpoint_sha_consistent"]:
            reasons.append("source_checkpoint_sha_mismatch")
        if not checkpoint_audit["checkpoint_boundary_passed"]:
            reasons.append("checkpoint_boundary_failed")
    return reasons


def _summary_counts(batch_audit: dict[str, Any], loss_gradient_audit: dict[str, Any], checkpoint_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: batch_audit[key] for key in (
            "batch_row_count",
            "slope_theta_batch_contract_missing_count",
            "xunce_batch_viewpoint_shape_mismatch_count",
            "action_viewpoint_binding_mismatch_count",
            "selected_info_viewpoint_mismatch_count",
            "slope_provenance_array_length_mismatch_count",
            "slope_provenance_binding_mismatch_count",
            "selected_action_mask_violation_count",
            "reward_fallback_used_count",
        )},
        "loss_row_count": loss_gradient_audit["loss_row_count"],
        "loss_finite": loss_gradient_audit["loss_finite"],
        "gradient_finite": loss_gradient_audit["gradient_finite"],
        "component_grad_norms_readable": loss_gradient_audit["component_grad_norms_readable"],
        "final_total_loss": loss_gradient_audit["final_total_loss"],
        "final_post_update_approx_kl": loss_gradient_audit["final_post_update_approx_kl"],
        "pre_clip_grad_norm": loss_gradient_audit["pre_clip_grad_norm"],
        "post_clip_grad_norm": loss_gradient_audit["post_clip_grad_norm"],
        "parameter_delta_l2": loss_gradient_audit["parameter_delta_l2"],
        "parameter_with_grad_count": loss_gradient_audit["parameter_with_grad_count"],
        "checkpoint_exists": checkpoint_audit["checkpoint_exists"],
        "checkpoint_reload_passed": checkpoint_audit["checkpoint_reload_passed"],
        "experimental_only": checkpoint_audit["experimental_only"],
        "experimental_checkpoint_sha256": checkpoint_audit["experimental_checkpoint_sha256"],
        "stage21_3_lineage_passed": checkpoint_audit["stage21_3_lineage_passed"],
        "checkpoint_path_in_stage21_4_root": checkpoint_audit["checkpoint_path_in_stage21_4_root"],
        "source_checkpoint_sha_consistent": checkpoint_audit["source_checkpoint_sha_consistent"],
        "checkpoint_boundary_passed": checkpoint_audit["checkpoint_boundary_passed"],
    }


def _input_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    summary_path = Path(config["stage23_3_root"]) / "xunce-stage23-3-summary.json"
    if not summary_path.is_file():
        return ["missing_stage23_3_summary"]
    summary = _read_json(summary_path)
    if summary.get("status") != "passed":
        reasons.append("stage23_3_not_passed")
    if summary.get("next_required_change") != ROUTE_FROM_STAGE23_3:
        reasons.append("stage23_3_route_not_stage23_4")
    if summary.get("coverage_source") != SLOPE_COVERAGE_SOURCE:
        reasons.append("stage23_3_coverage_source_not_slope_obstacle_los")
    if float(summary.get("max_traversable_slope_deg") or 0.0) != 30.0:
        reasons.append("stage23_3_max_traversable_slope_not_30")
    for key in (
        "stage21_1_source_roi_expansion_root_match",
        "stage21_3_rejects_old_reward_batch",
    ):
        if summary.get(key) is not True:
            reasons.append(f"stage23_3_{key}_not_true")
    for key in (
        "transition_id_count",
        "reward_transition_id_count",
        "batch_transition_id_count",
    ):
        if int(summary.get(key, 0) or 0) <= 0:
            reasons.append(f"stage23_3_{key}_missing")
    for key in (
        "mask_length_mismatch_count",
        "sampling_mask_contract_mismatch_count",
        "action_viewpoint_binding_mismatch_count",
        "slope_theta_transition_contract_missing_count",
        "strict_obstacle_aware_new_visible_false_count",
        "slope_theta_reward_contract_missing_count",
        "slope_theta_batch_reward_contract_missing_count",
        "reward_fallback_used_count",
        "reward_slope_footprint_binding_mismatch_count",
        "transition_missing_reward_count",
        "reward_transition_id_not_in_stage21_1_count",
        "batch_transition_id_not_in_stage21_1_count",
        "batch_transition_id_not_in_stage21_2_count",
        "transition_missing_batch_count",
        "reward_missing_batch_count",
        "hard_risk_violation_count",
        "mask_violation_count",
        "path_planning_failure_count",
        "open_grid_fallback_count",
    ):
        if int(summary.get(key, 0) or 0) != 0:
            reasons.append(f"stage23_3_{key}_nonzero")
    s21_3_root = Path(config["stage23_3_root"]) / "s21_3"
    for name in (
        "xunce-stage21-3-ppo-batch-validation-summary.json",
        "xunce-stage21-3-lineage-audit.json",
        "xunce-stage21-3-ppo-trainable-batch.jsonl",
    ):
        if not (s21_3_root / name).is_file():
            reasons.append(f"missing_stage23_3_s21_3_{name}")
    if not Path(config["xunce_candidate_checkpoint"]).is_file():
        reasons.append("missing_xunce_candidate_checkpoint")
    if not Path(config["high_fidelity_config"]).is_file():
        reasons.append("missing_high_fidelity_config")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FALSE_FIELDS if config.get(field) is True]
    if config.get("runs_new_ppo_update") is not True:
        reasons.append("runs_new_ppo_update_not_enabled_for_stage23_4_smoke")
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    defaults = {
        "stage23_3_root": DEFAULT_STAGE23_3_ROOT,
        "stage21_4_base_config": "configs/xunce_stage21_4_tiny_ppo_update_smoke_v1.json",
        "xunce_candidate_checkpoint": "outputs/path_feedback_batch_xunce_sandbox_candidate_preflight_v1/sandbox_package/xunce-controlled-training-candidate.pt",
        "high_fidelity_config": "configs/xunce_high_fidelity_exploration_coverage_comparison_stage18_9_strict_v3.json",
        "epochs": 1,
        "learning_rate": 2.0e-6,
        "clip_ratio": 0.2,
        "policy_loss_coefficient": 1.0,
        "value_loss_coefficient": 0.1,
        "entropy_coefficient": 0.01,
        "advantage_clip_abs": 5.0,
        "normalize_minibatch_advantages": True,
        "loss_scale": 0.25,
        "max_grad_norm": 1.0,
        "max_abs_approx_kl": 0.5,
        "stage23_4_authorized": False,
        "runs_new_ppo_update": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    for field in ("stage23_3_root", "stage21_4_base_config", "xunce_candidate_checkpoint", "high_fidelity_config"):
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    config["epochs"] = int(config["epochs"])
    for key in ("learning_rate", "clip_ratio", "policy_loss_coefficient", "value_loss_coefficient", "entropy_coefficient", "advantage_clip_abs", "loss_scale", "max_grad_norm", "max_abs_approx_kl", "canary_traffic_fraction"):
        config[key] = float(config[key])
    config["normalize_minibatch_advantages"] = bool(config["normalize_minibatch_advantages"])
    return config


def _tensor_second_dim_is(payload: Any, expected: int) -> bool:
    if not isinstance(payload, dict):
        return False
    shape = payload.get("shape")
    return isinstance(shape, list) and len(shape) >= 2 and int(shape[1]) == expected


def _mask_allows(info: dict[str, Any], mask_name: str, action_index: int | None) -> bool | None:
    if action_index is None or action_index < 0:
        return None
    mask = info.get(mask_name)
    if not isinstance(mask, list) or action_index >= len(mask):
        return None
    return mask[action_index] is True


def _theta_from_viewpoint(value: Any) -> int | None:
    if isinstance(value, list) and len(value) >= 3:
        return _int_or_none(value[2])
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _load_json_template(path: str, repo_root: Path) -> dict[str, Any]:
    return _read_json(_resolve_path(Path(path), repo_root))


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _same_resolved_path(value: Any, expected: Path) -> bool:
    if value is None:
        return False
    try:
        return Path(str(value)).resolve() == expected.resolve()
    except OSError:
        return str(value) == str(expected)


def _path_parent_is(value: Any, expected_parent: Path) -> bool:
    if value is None:
        return False
    try:
        return Path(str(value)).resolve().parent == expected_parent.resolve()
    except OSError:
        return False


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
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage23.4 Slope-Obstacle-Aware Theta PPO Update Smoke",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- batch_row_count: `{summary['batch_row_count']}`",
            f"- slope_theta_batch_contract_missing_count: `{summary['slope_theta_batch_contract_missing_count']}`",
            f"- xunce_batch_viewpoint_shape_mismatch_count: `{summary['xunce_batch_viewpoint_shape_mismatch_count']}`",
            f"- loss_finite: `{summary['loss_finite']}`",
            f"- gradient_finite: `{summary['gradient_finite']}`",
            f"- checkpoint_reload_passed: `{summary['checkpoint_reload_passed']}`",
            "",
            "Stage23.4 only validates the offline slope-obstacle-aware theta PPO update path. It does not run trajectory evaluation, publish checkpoints, replace default policy, connect executor, or start canary traffic.",
        ]
    ) + "\n"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
