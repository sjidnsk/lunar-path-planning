from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_EXPLORER_SRC = SCRIPT_DIR.parent / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

from model_explorer.policy.coverage_first_reward import (
    compute_coverage_first_reward_components,
    load_coverage_first_reward_profile,
)

try:  # pragma: no cover
    import run_xunce_stage21_2_coverage_first_ppo_reward_contract as stage21_2
    import run_xunce_stage21_3_ppo_batch_validation as stage21_3
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract as stage21_2
    import scripts.run_xunce_stage21_3_ppo_batch_validation as stage21_3


CONFIG_SCHEMA_VERSION = "xunce-stage23-2-slope-obstacle-aware-theta-reward-contract-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage23-2-summary/v1"
REPLAY_ROW_SCHEMA_VERSION = "xunce-stage23-2-reward-replay-row/v1"
CONTRACT_AUDIT_SCHEMA_VERSION = "xunce-stage23-2-reward-contract-audit/v1"
COMPAT_AUDIT_SCHEMA_VERSION = "xunce-stage23-2-stage21-compatibility-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage23-2-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage23-2-manifest/v1"

STAGE_ID = "xunce-stage23-2-slope-obstacle-aware-theta-reward-contract"
DEFAULT_CONFIG = "configs/xunce_stage23_2_slope_obstacle_aware_theta_reward_contract_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage23_slope_obstacle_aware_theta_reward/"
    "outputs/path_feedback_batch_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract_v1"
)
DEFAULT_STAGE23_2B_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage23_platform_alignment/"
    "outputs/path_feedback_batch_xunce_stage23_2b_platform_geometry_sensor_contract_alignment_v1"
)

SUMMARY_FILE = "xunce-stage23-2-summary.json"
CONTRACT_AUDIT_FILE = "xunce-stage23-2-reward-contract-audit.json"
REWARD_REPLAY_FILE = "xunce-stage23-2-reward-replay.jsonl"
COMPAT_AUDIT_FILE = "xunce-stage23-2-stage21-2-compatibility-audit.json"
BATCH_GATE_AUDIT_FILE = "xunce-stage23-2-stage21-3-batch-gate-audit.json"
ROUTING_FILE = "xunce-stage23-2-next-stage-routing.json"
REPORT_FILE = "xunce-stage23-2-report.md"
MANIFEST_FILE = "xunce-stage23-2-manifest.json"

LOS_AUDIT_FILE = "xunce-stage23-0-obstacle-los-audit.jsonl"
SLOPE_OBSTACLE_COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
ROUTE_INPUTS = "rerun_stage23_2_required_inputs"
ROUTE_PROVENANCE = "repair_stage23_2_reward_provenance"
ROUTE_DISCRIMINATION = "repair_stage23_2_reward_discrimination"
ROUTE_BATCH_GATE = "repair_stage23_2_batch_gate"
ROUTE_STAGE23_3 = "run_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke"
ROUTE_BOUNDARY = "resolve_stage23_2_boundary_rejections"

BOUNDARY_FIELDS = (
    "stage23_2_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage23.2 slope-obstacle-aware theta reward contract audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    profile = load_coverage_first_reward_profile(config["coverage_first_reward_profile"])
    boundary_reasons = _boundary_rejections(config)
    input_reasons = _input_rejections(config)
    los_rows = [] if input_reasons else _read_jsonl(Path(config["los_audit_path"]))

    replay_rows = _replay_rewards(los_rows, profile=profile, config=config)
    contract_audit = _contract_audit(replay_rows)
    compatibility_audit = _compatibility_audit(replay_rows, profile=profile, config=config)
    batch_gate_audit = compatibility_audit["stage21_3_batch_gate_audit"]
    status, route, route_reason = _route(boundary_reasons, input_reasons, contract_audit, compatibility_audit)
    blocking = _unique(boundary_reasons + input_reasons + _route_blocking_reasons(route, contract_audit, compatibility_audit))

    stage23_2b_summary = _read_json(Path(config["stage23_2b_root"]) / "xunce-stage23-2b-summary.json")
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking,
        "stage23_2b_root": config["stage23_2b_root"],
        "stage23_2b_status": stage23_2b_summary.get("status"),
        "stage23_2b_next_required_change": stage23_2b_summary.get("next_required_change"),
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "coverage_source": SLOPE_OBSTACLE_COVERAGE_SOURCE,
        "coverage_denominator_cells": float(config["coverage_denominator_cells"]),
        "platform_contract_id": stage23_2b_summary.get("platform_contract_id"),
        "platform_contract_hash": stage23_2b_summary.get("platform_contract_hash"),
        "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
        "reward_replay_row_count": len(replay_rows),
        "slope_obstacle_aware_theta_reward_contract_count": contract_audit[
            "slope_obstacle_aware_theta_reward_contract_count"
        ],
        "slope_obstacle_reward_contract_missing_count": contract_audit[
            "slope_obstacle_reward_contract_missing_count"
        ],
        "point_only_reward_fallback_used_count": contract_audit["point_only_reward_fallback_used_count"],
        "unobstructed_theta_reward_fallback_used_count": contract_audit["unobstructed_theta_reward_fallback_used_count"],
        "slope_los_changed_row_count": contract_audit["slope_los_changed_row_count"],
        "slope_los_changed_but_reward_equal_count": contract_audit["slope_los_changed_but_reward_equal_count"],
        "strict_obstacle_aware_new_visible_cell_count": contract_audit[
            "strict_obstacle_aware_new_visible_cell_count"
        ],
        "los_visible_count_proxy_used_count": contract_audit["los_visible_count_proxy_used_count"],
        "does_not_claim_collector_new_visible_provenance": contract_audit["los_visible_count_proxy_used_count"] > 0,
        "stage23_3_collector_new_visible_required": True,
        "stage21_2_slope_reward_supported": compatibility_audit["stage21_2_slope_reward_supported"],
        "stage21_3_accepts_slope_reward_batch": batch_gate_audit["accepts_slope_reward_batch"],
        "stage21_3_rejects_unobstructed_theta_reward_batch": batch_gate_audit["rejects_unobstructed_theta_reward_batch"],
        "stage21_3_rejects_point_only_reward_batch": batch_gate_audit["rejects_point_only_reward_batch"],
        "stage23_2_authorized": False,
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
        "stage23_2_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage23_2b_root": config["stage23_2b_root"],
        "los_audit_path": config["los_audit_path"],
        "summary": str(output_root / SUMMARY_FILE),
        "reward_contract_audit": str(output_root / CONTRACT_AUDIT_FILE),
        "reward_replay": str(output_root / REWARD_REPLAY_FILE),
        "stage21_2_compatibility_audit": str(output_root / COMPAT_AUDIT_FILE),
        "stage21_3_batch_gate_audit": str(output_root / BATCH_GATE_AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / CONTRACT_AUDIT_FILE, contract_audit)
    _write_jsonl(output_root / REWARD_REPLAY_FILE, replay_rows)
    _write_json(output_root / COMPAT_AUDIT_FILE, compatibility_audit)
    _write_json(output_root / BATCH_GATE_AUDIT_FILE, batch_gate_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _replay_rewards(rows: list[dict[str, Any]], *, profile: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    replay_rows: list[dict[str, Any]] = []
    denominator = float(config["coverage_denominator_cells"])
    max_rows = int(config["max_replay_rows"])
    platform_hash = str(config["platform_contract_hash"])
    max_slope = float(config["max_traversable_slope_deg"])
    for row in rows[:max_rows]:
        obstacle_new = _finite(row.get("obstacle_aware_new_visible_cell_count"))
        obstacle_new_source = "obstacle_aware_new_visible_cell_count"
        if obstacle_new is None:
            obstacle_new = _finite(row.get("obstacle_aware_visible_cell_count"))
            obstacle_new_source = (
                "stage23_0_obstacle_aware_visible_cell_count_replay_proxy"
                if obstacle_new is not None
                else "missing"
            )
        clear_new = _finite(row.get("clear_visible_cell_count"))
        path_cost = _positive_or_default(row.get("path_cost"), 1.0)
        slope_cpc = (float(obstacle_new) / path_cost) if obstacle_new is not None else None
        metrics = _reward_metrics(obstacle_new, slope_cpc, denominator=denominator, path_cost=path_cost)
        clear_metrics = _reward_metrics(clear_new, (float(clear_new) / path_cost) if clear_new is not None else None, denominator=denominator, path_cost=path_cost)
        result = compute_coverage_first_reward_components(metrics, profile)
        clear_result = compute_coverage_first_reward_components(clear_metrics, profile)
        replay_rows.append(
            {
                "schema_version": REPLAY_ROW_SCHEMA_VERSION,
                "scenario_id": row.get("scenario_id"),
                "step_index": row.get("step_index"),
                "candidate_index": row.get("candidate_index"),
                "candidate_cell": row.get("candidate_cell"),
                "candidate_viewpoint": row.get("candidate_viewpoint"),
                "candidate_theta_deg": row.get("candidate_theta_deg"),
                "sensor_fov_deg": row.get("sensor_fov_deg"),
                "sensor_range_cells": row.get("sensor_range_cells"),
                "theta_aware_reward_contract": True,
                "slope_obstacle_aware_theta_reward_contract": True,
                "coverage_source": SLOPE_OBSTACLE_COVERAGE_SOURCE,
                "clear_visible_cell_count": clear_new,
                "clear_theta_coverage_hash": row.get("clear_theta_coverage_hash"),
                "clear_unobstructed_reward": clear_result.reward,
                "obstacle_aware_new_visible_cell_count": obstacle_new,
                "obstacle_aware_new_visible_cell_count_source": obstacle_new_source,
                "strict_obstacle_aware_new_visible_cell_count": (
                    obstacle_new_source == "obstacle_aware_new_visible_cell_count"
                ),
                "obstacle_aware_theta_coverage_hash": row.get("obstacle_aware_theta_coverage_hash"),
                "obstacle_aware_theta_coverage_gain_per_path_cost": slope_cpc,
                "obstacle_aware_theta_coverage_denominator_cells": denominator,
                "visibility_removed_by_obstacle_count": row.get("visibility_removed_by_obstacle_count"),
                "slope_obstacle_source_hash": row.get("obstacle_source_hash"),
                "platform_contract_hash": platform_hash,
                "max_traversable_slope_deg": max_slope,
                "slope_blocked_source_kind": row.get("obstacle_source_kind"),
                "point_only_reward_fallback_used": False,
                "unobstructed_theta_reward_fallback_used": False,
                "metrics": metrics,
                "reward": result.reward,
                "components": result.components,
                "trainable": result.trainable,
                "reason_codes": result.reason_codes,
                "profile_id": result.profile_id,
                "profile_version": result.profile_version,
                "profile_hash": result.profile_hash,
                "reward_discriminates_slope_los": _different(clear_result.reward, result.reward),
            }
        )
    return replay_rows


def _reward_metrics(coverage_count: float | None, cpc: float | None, *, denominator: float, path_cost: float) -> dict[str, Any]:
    coverage_rate = (float(coverage_count) / denominator) if coverage_count is not None else None
    return {
        "coverage_rate_delta": coverage_rate,
        "coverage_per_cost": cpc,
        "coverage_progress_rate": coverage_rate,
        "final_coverage_rate": coverage_rate,
        "path_cost_m": path_cost,
        "soft_risk_exposure": 0.0,
        "done": False,
        "hard_risk_flags": [],
        "hard_risk_violation_count": 0,
        "open_grid_fallback_used": False,
    }


def _contract_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    changed_rows = [
        row
        for row in rows
        if _finite(row.get("clear_visible_cell_count")) != _finite(row.get("obstacle_aware_new_visible_cell_count"))
    ]
    equal_changed = [row for row in changed_rows if not row.get("reward_discriminates_slope_los")]
    examples = [
        {
            "scenario_id": row.get("scenario_id"),
            "step_index": row.get("step_index"),
            "candidate_viewpoint": row.get("candidate_viewpoint"),
            "clear_visible_cell_count": row.get("clear_visible_cell_count"),
            "obstacle_aware_new_visible_cell_count": row.get("obstacle_aware_new_visible_cell_count"),
            "clear_unobstructed_reward": row.get("clear_unobstructed_reward"),
            "slope_obstacle_reward": row.get("reward"),
        }
        for row in changed_rows[:5]
    ]
    return {
        "schema_version": CONTRACT_AUDIT_SCHEMA_VERSION,
        "reward_replay_row_count": len(rows),
        "slope_obstacle_aware_theta_reward_contract_count": sum(
            1 for row in rows if _row_has_slope_obstacle_reward_contract(row)
        ),
        "slope_obstacle_reward_contract_missing_count": sum(
            1 for row in rows if not _row_has_slope_obstacle_reward_contract(row)
        ),
        "point_only_reward_fallback_used_count": sum(1 for row in rows if row.get("point_only_reward_fallback_used") is True),
        "unobstructed_theta_reward_fallback_used_count": sum(
            1 for row in rows if row.get("unobstructed_theta_reward_fallback_used") is True
        ),
        "slope_los_changed_row_count": len(changed_rows),
        "slope_los_changed_but_reward_equal_count": len(equal_changed),
        "strict_obstacle_aware_new_visible_cell_count": sum(
            1 for row in rows if row.get("strict_obstacle_aware_new_visible_cell_count") is True
        ),
        "los_visible_count_proxy_used_count": sum(
            1
            for row in rows
            if row.get("obstacle_aware_new_visible_cell_count_source")
            == "stage23_0_obstacle_aware_visible_cell_count_replay_proxy"
        ),
        "reward_discrimination_examples": examples,
    }


def _compatibility_audit(rows: list[dict[str, Any]], *, profile: Any, config: dict[str, Any]) -> dict[str, Any]:
    synthetic: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:2]):
        synthetic.append(
            {
                "transition_id": f"slope-sample:{index}",
                "scenario_id": row.get("scenario_id") or "s",
                "step_index": index,
                "done": False,
                "trainable": True,
                "action_index": 0,
                "info": {
                    "candidate_viewpoints": [row.get("candidate_viewpoint")],
                    "candidate_theta_deg": [row.get("candidate_theta_deg")],
                    "theta_new_visible_cell_counts": [row.get("clear_visible_cell_count")],
                    "theta_coverage_hashes": [row.get("clear_theta_coverage_hash") or "clear-hash"],
                    "theta_coverage_gain_per_path_costs": [row.get("clear_visible_cell_count") or 0.0],
                    "obstacle_aware_new_visible_cell_counts": [row.get("obstacle_aware_new_visible_cell_count")],
                    "obstacle_aware_theta_coverage_hashes": [row.get("obstacle_aware_theta_coverage_hash")],
                    "obstacle_aware_theta_coverage_gain_per_path_costs": [
                        row.get("obstacle_aware_theta_coverage_gain_per_path_cost")
                    ],
                    "slope_obstacle_source_hash": row.get("slope_obstacle_source_hash"),
                    "platform_contract_hash": row.get("platform_contract_hash"),
                    "max_traversable_slope_deg": row.get("max_traversable_slope_deg"),
                    "slope_blocked_source_kind": row.get("slope_blocked_source_kind"),
                    "final_coverage_rate_after_step": 0.1,
                    "path_cost": 1.0,
                    "soft_risk_exposure": 0.0,
                    "hard_risk_violation": False,
                },
            }
        )
    stage21_2_rows = stage21_2._evaluate_transitions(
        synthetic,
        profile=profile,
        config={
            "require_theta_aware_reward_contract": True,
            "slope_obstacle_aware_theta_reward_enabled": True,
            "theta_coverage_denominator_cells": float(config["coverage_denominator_cells"]),
        },
    )
    slope_supported = bool(stage21_2_rows) and all(_row_has_slope_obstacle_reward_contract(row) for row in stage21_2_rows)
    batch_rows = []
    for reward_row, transition in zip(stage21_2_rows, synthetic):
        batch_like = dict(reward_row)
        batch_like["info"] = transition.get("info")
        batch_like["action_index"] = transition.get("action_index")
        batch_like["reward_metrics"] = reward_row.get("metrics")
        batch_rows.append(batch_like)
    accept_missing = stage21_3._slope_obstacle_reward_contract_missing_count(batch_rows)
    unobstructed = [dict(row) for row in batch_rows[:1]]
    if unobstructed:
        unobstructed[0]["coverage_source"] = "theta_aware_sensor_footprint/v1"
        unobstructed[0]["slope_obstacle_aware_theta_reward_contract"] = False
        unobstructed[0]["unobstructed_theta_reward_fallback_used"] = True
    point_only = [{"reward": 1.0}]
    batch_gate = {
        "schema_version": "xunce-stage23-2-stage21-3-batch-gate-audit/v1",
        "accepts_slope_reward_batch": accept_missing == 0 and bool(batch_rows),
        "rejects_unobstructed_theta_reward_batch": stage21_3._slope_obstacle_reward_contract_missing_count(unobstructed)
        == len(unobstructed)
        and bool(unobstructed),
        "rejects_point_only_reward_batch": stage21_3._slope_obstacle_reward_contract_missing_count(point_only) == 1,
        "slope_reward_missing_count_for_slope_rows": accept_missing,
    }
    return {
        "schema_version": COMPAT_AUDIT_SCHEMA_VERSION,
        "stage21_2_slope_reward_supported": slope_supported,
        "stage21_2_slope_reward_row_count": len(stage21_2_rows),
        "stage21_2_slope_reward_coverage_source_count": sum(
            1 for row in stage21_2_rows if row.get("coverage_source") == SLOPE_OBSTACLE_COVERAGE_SOURCE
        ),
        "stage21_3_batch_gate_audit": batch_gate,
    }


def _route(
    boundary_reasons: list[str],
    input_reasons: list[str],
    contract: dict[str, Any],
    compatibility: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage23_2_boundary_rejected"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage23_2_inputs_missing_or_untrusted"
    if contract["slope_obstacle_reward_contract_missing_count"] or contract["point_only_reward_fallback_used_count"]:
        return "failed", ROUTE_PROVENANCE, "slope_obstacle_reward_provenance_missing"
    if contract["unobstructed_theta_reward_fallback_used_count"]:
        return "failed", ROUTE_PROVENANCE, "unobstructed_theta_reward_fallback_used"
    if contract["slope_los_changed_row_count"] and contract["slope_los_changed_but_reward_equal_count"]:
        return "failed", ROUTE_DISCRIMINATION, "slope_los_reward_not_discriminated"
    batch_gate = compatibility["stage21_3_batch_gate_audit"]
    if not compatibility["stage21_2_slope_reward_supported"]:
        return "failed", ROUTE_PROVENANCE, "stage21_2_slope_reward_not_supported"
    if not (
        batch_gate["accepts_slope_reward_batch"]
        and batch_gate["rejects_unobstructed_theta_reward_batch"]
        and batch_gate["rejects_point_only_reward_batch"]
    ):
        return "failed", ROUTE_BATCH_GATE, "stage21_3_slope_reward_gate_missing"
    return "passed", ROUTE_STAGE23_3, "slope_obstacle_aware_theta_reward_contract_ready"


def _route_blocking_reasons(route: str, contract: dict[str, Any], compatibility: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if route == ROUTE_PROVENANCE:
        if contract.get("slope_obstacle_reward_contract_missing_count"):
            reasons.append("slope_obstacle_reward_contract_missing")
        if contract.get("point_only_reward_fallback_used_count"):
            reasons.append("point_only_reward_fallback_used")
        if contract.get("unobstructed_theta_reward_fallback_used_count"):
            reasons.append("unobstructed_theta_reward_fallback_used")
        if not compatibility.get("stage21_2_slope_reward_supported"):
            reasons.append("stage21_2_slope_reward_not_supported")
    if route == ROUTE_DISCRIMINATION:
        reasons.append("slope_los_changed_but_reward_equal")
    if route == ROUTE_BATCH_GATE:
        reasons.append("stage21_3_slope_reward_gate_missing")
    return reasons


def _input_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    root = Path(config["stage23_2b_root"])
    summary_path = root / "xunce-stage23-2b-summary.json"
    if not summary_path.is_file():
        return ["missing_stage23_2b_summary"]
    summary = _read_json(summary_path)
    if summary.get("status") != "passed":
        reasons.append("stage23_2b_not_passed")
    if summary.get("next_required_change") != "implement_stage23_2_slope_obstacle_aware_theta_reward_contract":
        reasons.append("stage23_2b_route_not_stage23_2")
    if float(summary.get("max_traversable_slope_deg", 0.0) or 0.0) != float(config["max_traversable_slope_deg"]):
        reasons.append("stage23_2b_max_slope_mismatch")
    if summary.get("platform_contract_hash") != config.get("platform_contract_hash"):
        reasons.append("platform_contract_hash_mismatch")
    if not Path(config["los_audit_path"]).is_file():
        reasons.append("missing_stage23_0_los_audit")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [f"{field}_enabled" for field in BOUNDARY_FIELDS if config.get(field) is not False]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction_nonzero")
    return reasons


def _load_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(config_path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported config schema_version: {payload.get('schema_version')}")
    defaults = {
        "stage23_2b_root": DEFAULT_STAGE23_2B_ROOT,
        "coverage_first_reward_profile": "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json",
        "coverage_denominator_cells": 100.0,
        "max_replay_rows": 5000,
        "max_traversable_slope_deg": 30.0,
        "stage23_2_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    config["stage23_2b_root"] = str(_resolve_path(Path(str(config["stage23_2b_root"])), repo_root))
    stage23_2b_root = Path(config["stage23_2b_root"])
    summary = _read_json(stage23_2b_root / "xunce-stage23-2b-summary.json") if (stage23_2b_root / "xunce-stage23-2b-summary.json").exists() else {}
    config.setdefault("platform_contract_hash", summary.get("platform_contract_hash"))
    config["platform_contract_hash"] = str(config.get("platform_contract_hash") or "")
    config["coverage_first_reward_profile"] = str(_resolve_path(Path(str(config["coverage_first_reward_profile"])), repo_root))
    config["coverage_denominator_cells"] = _positive_float(config["coverage_denominator_cells"], "coverage_denominator_cells")
    config["max_replay_rows"] = _positive_int(config["max_replay_rows"], "max_replay_rows")
    config["max_traversable_slope_deg"] = _positive_float(config["max_traversable_slope_deg"], "max_traversable_slope_deg")
    if "los_audit_path" not in config:
        discovered = _discover_los_audit_path(stage23_2b_root)
        config["los_audit_path"] = str(
            discovered
            if discovered is not None
            else stage23_2b_root / "s23_2a_platform_30" / "s23_1" / "s23_1_a" / "s23_0" / LOS_AUDIT_FILE
        )
    else:
        config["los_audit_path"] = str(_resolve_path(Path(str(config["los_audit_path"])), repo_root))
    return config


def _discover_los_audit_path(stage23_2b_root: Path) -> Path | None:
    if not stage23_2b_root.is_dir():
        return None
    candidates = sorted(stage23_2b_root.rglob(LOS_AUDIT_FILE), key=lambda path: str(path))
    if not candidates:
        return None
    for path in candidates:
        parts = {part.lower() for part in path.parts}
        if "a30" in parts or "s23_2a_platform_30" in parts:
            return path
    return candidates[0]


def _row_has_slope_obstacle_reward_contract(row: dict[str, Any]) -> bool:
    return (
        row.get("theta_aware_reward_contract") is True
        and row.get("slope_obstacle_aware_theta_reward_contract") is True
        and row.get("coverage_source") == SLOPE_OBSTACLE_COVERAGE_SOURCE
        and isinstance(row.get("candidate_viewpoint"), list)
        and len(row.get("candidate_viewpoint")) >= 3
        and _finite(row.get("candidate_theta_deg")) is not None
        and _finite(row.get("obstacle_aware_new_visible_cell_count")) is not None
        and _finite(row.get("obstacle_aware_theta_coverage_gain_per_path_cost")) is not None
        and _finite(row.get("obstacle_aware_theta_coverage_denominator_cells")) is not None
        and isinstance(row.get("obstacle_aware_theta_coverage_hash"), str)
        and bool(str(row.get("obstacle_aware_theta_coverage_hash")).strip())
        and isinstance(row.get("slope_obstacle_source_hash"), str)
        and bool(str(row.get("slope_obstacle_source_hash")).strip())
        and isinstance(row.get("platform_contract_hash"), str)
        and bool(str(row.get("platform_contract_hash")).strip())
        and _finite(row.get("max_traversable_slope_deg")) == 30.0
        and row.get("slope_blocked_source_kind") == "slope_blocked_as_obstacle_proxy"
        and row.get("point_only_reward_fallback_used") is False
        and row.get("unobstructed_theta_reward_fallback_used") is False
    )


def _different(left: Any, right: Any) -> bool:
    left_value = _finite(left)
    right_value = _finite(right)
    if left_value is None or right_value is None:
        return False
    return abs(left_value - right_value) > 1.0e-12


def _positive_or_default(value: Any, default: float) -> float:
    number = _finite(value)
    return float(number) if number is not None and number > 0.0 else float(default)


def _positive_float(value: Any, name: str) -> float:
    number = _finite(value)
    if number is None or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return float(number)


def _positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number <= 0:
        raise ValueError(f"{name} must be > 0")
    return number


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage23.2 Slope-Obstacle-Aware Theta Reward Contract",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- coverage_source: `{summary['coverage_source']}`",
            f"- max_traversable_slope_deg: `{summary['max_traversable_slope_deg']}`",
            f"- reward_replay_row_count: `{summary['reward_replay_row_count']}`",
            "- slope_obstacle_reward_contract_missing_count: "
            f"`{summary['slope_obstacle_reward_contract_missing_count']}`",
            f"- slope_los_changed_row_count: `{summary['slope_los_changed_row_count']}`",
            f"- los_visible_count_proxy_used_count: `{summary['los_visible_count_proxy_used_count']}`",
            "- stage23_3_collector_new_visible_required: "
            f"`{summary['stage23_3_collector_new_visible_required']}`",
            "",
            "Stage23.2 validates reward provenance only. LOS replay rows may use Stage23.0 visible-count proxy because the real collector new-visible source is checked in Stage23.3. It does not run PPO, publish checkpoints, replace default policy, connect executor, or start canary traffic.",
        ]
    ) + "\n"


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
