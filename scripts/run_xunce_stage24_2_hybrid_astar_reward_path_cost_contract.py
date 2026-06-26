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


CONFIG_SCHEMA_VERSION = "xunce-stage24-2-hybrid-astar-reward-path-cost-contract-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage24-2-summary/v1"
REPLAY_ROW_SCHEMA_VERSION = "xunce-stage24-2-reward-replay-row/v1"
CONTRACT_AUDIT_SCHEMA_VERSION = "xunce-stage24-2-reward-path-cost-contract-audit/v1"
COMPAT_AUDIT_SCHEMA_VERSION = "xunce-stage24-2-stage21-compatibility-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage24-2-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage24-2-manifest/v1"

STAGE_ID = "xunce-stage24-2-hybrid-astar-reward-path-cost-contract"
DEFAULT_CONFIG = "configs/xunce_stage24_2_hybrid_astar_reward_path_cost_contract_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage24_hybrid_astar_pose_planner/"
    "outputs/path_feedback_batch_xunce_stage24_2_hybrid_astar_reward_path_cost_contract_v1"
)
DEFAULT_STAGE24_1_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage24_hybrid_astar_pose_planner/"
    "outputs/path_feedback_batch_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration_v1"
)

SUMMARY_FILE = "xunce-stage24-2-summary.json"
CONTRACT_AUDIT_FILE = "xunce-stage24-2-reward-path-cost-contract-audit.json"
REWARD_REPLAY_FILE = "xunce-stage24-2-reward-replay.jsonl"
COMPAT_AUDIT_FILE = "xunce-stage24-2-stage21-2-compatibility-audit.json"
BATCH_GATE_AUDIT_FILE = "xunce-stage24-2-stage21-3-batch-gate-audit.json"
ROUTING_FILE = "xunce-stage24-2-next-stage-routing.json"
REPORT_FILE = "xunce-stage24-2-report.md"
MANIFEST_FILE = "xunce-stage24-2-manifest.json"

STAGE24_1_SUMMARY_FILE = "xunce-stage24-1-summary.json"
STAGE24_1_AUDIT_FILE = "xunce-stage24-1-candidate-hybrid-path-cost-audit.jsonl"
STAGE24_1_RECOMMENDATION_FILE = "xunce-stage24-1-path-cost-source-recommendation.json"
HYBRID_ASTAR_PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"
SLOPE_OBSTACLE_COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"

ROUTE_INPUTS = "rerun_stage24_2_required_inputs"
ROUTE_SOURCE = "repair_stage24_2_reward_path_cost_source"
ROUTE_DISCRIMINATION = "repair_stage24_2_reward_discrimination"
ROUTE_BATCH_GATE = "repair_stage24_2_batch_gate"
ROUTE_STAGE24_3 = "run_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke"
ROUTE_BOUNDARY = "resolve_stage24_2_boundary_rejections"

BOUNDARY_FIELDS = (
    "stage24_2_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage24.2 Hybrid A* reward path-cost contract audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage24_2_hybrid_astar_reward_path_cost_contract(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage24_2_hybrid_astar_reward_path_cost_contract(
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
    candidate_rows = [] if input_reasons else _read_jsonl(Path(config["candidate_audit_path"]))

    replay_rows = _replay_rewards(candidate_rows, profile=profile, config=config)
    contract_audit = _contract_audit(replay_rows)
    compatibility_audit = _compatibility_audit(replay_rows, profile=profile, config=config)
    batch_gate_audit = compatibility_audit["stage21_3_batch_gate_audit"]
    status, route, route_reason = _route(boundary_reasons, input_reasons, contract_audit, compatibility_audit)
    blocking = _unique(boundary_reasons + input_reasons + _route_blocking_reasons(route, contract_audit, compatibility_audit))

    stage24_1_summary_path = Path(config["stage24_1_root"]) / STAGE24_1_SUMMARY_FILE
    stage24_1_summary = _read_json(stage24_1_summary_path) if stage24_1_summary_path.is_file() else {}
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking,
        "stage24_1_root": config["stage24_1_root"],
        "stage24_1_status": stage24_1_summary.get("status"),
        "stage24_1_next_required_change": stage24_1_summary.get("next_required_change"),
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "coverage_source": SLOPE_OBSTACLE_COVERAGE_SOURCE,
        "path_cost_source": HYBRID_ASTAR_PATH_COST_SOURCE,
        "path_cost_source_recommendation": HYBRID_ASTAR_PATH_COST_SOURCE,
        "coverage_denominator_cells": float(config["coverage_denominator_cells"]),
        "platform_contract_hash": stage24_1_summary.get("platform_contract_hash"),
        "max_traversable_slope_deg": float(stage24_1_summary.get("max_traversable_slope_deg") or 30.0),
        "reward_replay_row_count": len(replay_rows),
        "hybrid_astar_path_cost_reward_contract_count": contract_audit[
            "hybrid_astar_path_cost_reward_contract_count"
        ],
        "hybrid_astar_path_cost_contract_missing_count": contract_audit[
            "hybrid_astar_path_cost_contract_missing_count"
        ],
        "point_grid_path_cost_fallback_used_count": contract_audit["point_grid_path_cost_fallback_used_count"],
        "slope_blocked_source_kind_inferred_from_stage24_2_config_count": contract_audit[
            "slope_blocked_source_kind_inferred_from_stage24_2_config_count"
        ],
        "hybrid_grid_cost_material_row_count": contract_audit["hybrid_grid_cost_material_row_count"],
        "hybrid_grid_cost_diff_but_reward_equal_count": contract_audit[
            "hybrid_grid_cost_diff_but_reward_equal_count"
        ],
        "default_astar_replaced": False,
        "ackermann_feasible_claimed": False,
        "stage21_2_hybrid_path_cost_supported": compatibility_audit["stage21_2_hybrid_path_cost_supported"],
        "stage21_3_accepts_hybrid_path_cost_batch": batch_gate_audit["accepts_hybrid_path_cost_batch"],
        "stage21_3_rejects_grid_only_path_cost_batch": batch_gate_audit["rejects_grid_only_path_cost_batch"],
        "stage21_3_rejects_ackermann_claim_batch": batch_gate_audit["rejects_ackermann_claim_batch"],
        "stage24_2_authorized": False,
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
        "stage24_2_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage24_1_root": config["stage24_1_root"],
        "candidate_audit_path": config["candidate_audit_path"],
        "summary": str(output_root / SUMMARY_FILE),
        "reward_path_cost_contract_audit": str(output_root / CONTRACT_AUDIT_FILE),
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
    default_new_visible = float(config["replay_obstacle_aware_new_visible_cell_count"])
    for row in rows[:max_rows]:
        if row.get("hybrid_astar_reachable") is not True:
            continue
        obstacle_new = _finite(row.get("obstacle_aware_new_visible_cell_count"))
        if obstacle_new is None:
            obstacle_new = default_new_visible
        hybrid_cost = _finite(row.get("hybrid_astar_path_cost"))
        grid_cost = _finite(row.get("legacy_grid_astar_path_cost") or row.get("path_cost"))
        slope_source_kind = row.get("slope_blocked_source_kind") or config["slope_blocked_source_kind"]
        metrics = _reward_metrics(obstacle_new, denominator=denominator, path_cost=hybrid_cost)
        grid_metrics = _reward_metrics(obstacle_new, denominator=denominator, path_cost=grid_cost)
        result = compute_coverage_first_reward_components(metrics, profile)
        grid_result = compute_coverage_first_reward_components(grid_metrics, profile)
        material_delta = _different(row.get("hybrid_vs_grid_path_cost_delta"), 0.0)
        replay_rows.append(
            {
                "schema_version": REPLAY_ROW_SCHEMA_VERSION,
                "scenario_id": row.get("scenario_id"),
                "step_index": row.get("step_index"),
                "candidate_index": row.get("candidate_index"),
                "candidate_cell": row.get("candidate_cell"),
                "candidate_viewpoint": row.get("candidate_viewpoint"),
                "candidate_theta_deg": row.get("candidate_theta_deg"),
                "theta_aware_reward_contract": True,
                "slope_obstacle_aware_theta_reward_contract": True,
                "coverage_source": SLOPE_OBSTACLE_COVERAGE_SOURCE,
                "obstacle_aware_new_visible_cell_count": int(obstacle_new),
                "obstacle_aware_theta_coverage_hash": row.get("obstacle_aware_theta_coverage_hash")
                or f"stage24-2-obstacle-aware-los-{row.get('candidate_index')}",
                "obstacle_aware_theta_coverage_gain_per_path_cost": _coverage_per_cost(obstacle_new, hybrid_cost),
                "obstacle_aware_theta_coverage_denominator_cells": denominator,
                "slope_obstacle_source_hash": row.get("slope_obstacle_source_hash"),
                "platform_contract_hash": row.get("platform_contract_hash"),
                "max_traversable_slope_deg": row.get("max_traversable_slope_deg"),
                "slope_blocked_source_kind": slope_source_kind,
                "slope_blocked_source_kind_inferred_from_stage24_2_config": row.get("slope_blocked_source_kind") is None,
                "hybrid_astar_path_cost_reward_contract": _row_has_hybrid_path_source(row),
                "path_cost_source": HYBRID_ASTAR_PATH_COST_SOURCE if _row_has_hybrid_path_source(row) else row.get("path_cost_source"),
                "hybrid_astar_path_cost": hybrid_cost,
                "hybrid_astar_pose_path_hash": row.get("hybrid_astar_pose_path_hash"),
                "hybrid_astar_trajectory_kind": row.get("hybrid_astar_trajectory_kind"),
                "legacy_grid_astar_path_cost": grid_cost,
                "hybrid_vs_grid_path_cost_delta": row.get("hybrid_vs_grid_path_cost_delta"),
                "default_astar_replaced": row.get("default_astar_replaced"),
                "hybrid_astar_ackermann_feasible_claimed": row.get("hybrid_astar_ackermann_feasible_claimed"),
                "point_grid_path_cost_fallback_used": not _row_has_hybrid_path_source(row),
                "point_only_reward_fallback_used": False,
                "unobstructed_theta_reward_fallback_used": False,
                "metrics": metrics,
                "grid_cost_metrics": grid_metrics,
                "reward": result.reward,
                "components": result.components,
                "legacy_grid_astar_reward": grid_result.reward,
                "legacy_grid_astar_components": grid_result.components,
                "hybrid_grid_cost_material": material_delta,
                "reward_discriminates_hybrid_cost": material_delta
                and _reward_or_components_different(result, grid_result),
                "trainable": result.trainable,
                "reason_codes": result.reason_codes,
                "profile_id": result.profile_id,
                "profile_version": result.profile_version,
                "profile_hash": result.profile_hash,
            }
        )
    return replay_rows


def _reward_metrics(coverage_count: float | None, *, denominator: float, path_cost: float | None) -> dict[str, Any]:
    coverage_rate = (float(coverage_count) / denominator) if coverage_count is not None else None
    cpc = _coverage_per_cost(coverage_count, path_cost)
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
    material = [row for row in rows if row.get("hybrid_grid_cost_material") is True]
    equal_material = [row for row in material if row.get("reward_discriminates_hybrid_cost") is not True]
    return {
        "schema_version": CONTRACT_AUDIT_SCHEMA_VERSION,
        "reward_replay_row_count": len(rows),
        "hybrid_astar_path_cost_reward_contract_count": sum(1 for row in rows if _row_has_hybrid_reward_contract(row)),
        "hybrid_astar_path_cost_contract_missing_count": sum(
            1 for row in rows if not _row_has_hybrid_reward_contract(row)
        ),
        "point_grid_path_cost_fallback_used_count": sum(1 for row in rows if row.get("point_grid_path_cost_fallback_used") is True),
        "slope_blocked_source_kind_inferred_from_stage24_2_config_count": sum(
            1 for row in rows if row.get("slope_blocked_source_kind_inferred_from_stage24_2_config") is True
        ),
        "hybrid_grid_cost_material_row_count": len(material),
        "hybrid_grid_cost_diff_but_reward_equal_count": len(equal_material),
        "hybrid_grid_cost_discrimination_examples": [
            {
                "scenario_id": row.get("scenario_id"),
                "step_index": row.get("step_index"),
                "candidate_viewpoint": row.get("candidate_viewpoint"),
                "hybrid_astar_path_cost": row.get("hybrid_astar_path_cost"),
                "legacy_grid_astar_path_cost": row.get("legacy_grid_astar_path_cost"),
                "hybrid_reward": row.get("reward"),
                "legacy_grid_reward": row.get("legacy_grid_astar_reward"),
            }
            for row in material[:5]
        ],
    }


def _compatibility_audit(rows: list[dict[str, Any]], *, profile: Any, config: dict[str, Any]) -> dict[str, Any]:
    synthetic: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:2]):
        synthetic.append(_synthetic_transition(row, index=index, mutate=None))
    stage21_2_rows = stage21_2._evaluate_transitions(
        synthetic,
        profile=profile,
        config={
            "require_theta_aware_reward_contract": True,
            "slope_obstacle_aware_theta_reward_enabled": True,
            "require_hybrid_astar_path_cost_contract": True,
            "theta_coverage_denominator_cells": float(config["coverage_denominator_cells"]),
        },
    )
    hybrid_supported = bool(stage21_2_rows) and all(_row_has_hybrid_reward_contract(row) for row in stage21_2_rows)
    batch_rows = _batch_like(stage21_2_rows, synthetic)
    grid_only = _batch_like(
        [
            _mutated_reward_row(row, path_cost_source="legacy_grid_astar_path/v1", contract=False, fallback=True)
            for row in stage21_2_rows[:1]
        ],
        synthetic[:1],
    )
    ackermann = _batch_like(
        [
            _mutated_reward_row(row, hybrid_astar_ackermann_feasible_claimed=True, contract=False)
            for row in stage21_2_rows[:1]
        ],
        synthetic[:1],
    )
    batch_gate = {
        "schema_version": "xunce-stage24-2-stage21-3-batch-gate-audit/v1",
        "accepts_hybrid_path_cost_batch": stage21_3._hybrid_astar_path_cost_contract_missing_count(batch_rows) == 0
        and bool(batch_rows),
        "rejects_grid_only_path_cost_batch": stage21_3._hybrid_astar_path_cost_contract_missing_count(grid_only)
        == len(grid_only)
        and bool(grid_only),
        "rejects_ackermann_claim_batch": stage21_3._hybrid_astar_path_cost_contract_missing_count(ackermann)
        == len(ackermann)
        and bool(ackermann),
        "hybrid_path_cost_missing_count_for_hybrid_rows": stage21_3._hybrid_astar_path_cost_contract_missing_count(
            batch_rows
        ),
    }
    return {
        "schema_version": COMPAT_AUDIT_SCHEMA_VERSION,
        "stage21_2_hybrid_path_cost_supported": hybrid_supported,
        "stage21_2_hybrid_path_cost_row_count": len(stage21_2_rows),
        "stage21_2_hybrid_path_cost_source_count": sum(
            1 for row in stage21_2_rows if row.get("path_cost_source") == HYBRID_ASTAR_PATH_COST_SOURCE
        ),
        "stage21_3_batch_gate_audit": batch_gate,
    }


def _synthetic_transition(row: dict[str, Any], *, index: int, mutate: dict[str, Any] | None) -> dict[str, Any]:
    payload = {
        "transition_id": f"hybrid-sample:{index}",
        "scenario_id": row.get("scenario_id") or "s",
        "step_index": index,
        "done": False,
        "trainable": True,
        "action_index": 0,
        "info": {
            "candidate_viewpoints": [row.get("candidate_viewpoint")],
            "candidate_theta_deg": [row.get("candidate_theta_deg")],
            "theta_new_visible_cell_counts": [row.get("obstacle_aware_new_visible_cell_count")],
            "theta_coverage_hashes": [row.get("obstacle_aware_theta_coverage_hash") or "theta-hash"],
            "theta_coverage_gain_per_path_costs": [row.get("obstacle_aware_theta_coverage_gain_per_path_cost")],
            "obstacle_aware_new_visible_cell_counts": [row.get("obstacle_aware_new_visible_cell_count")],
            "obstacle_aware_theta_coverage_hashes": [row.get("obstacle_aware_theta_coverage_hash")],
            "obstacle_aware_theta_coverage_gain_per_path_costs": [
                row.get("obstacle_aware_theta_coverage_gain_per_path_cost")
            ],
            "slope_obstacle_source_hash": row.get("slope_obstacle_source_hash"),
            "platform_contract_hash": row.get("platform_contract_hash"),
            "max_traversable_slope_deg": row.get("max_traversable_slope_deg"),
            "slope_blocked_source_kind": row.get("slope_blocked_source_kind"),
            "path_cost_source": row.get("path_cost_source"),
            "hybrid_astar_path_cost": row.get("hybrid_astar_path_cost"),
            "hybrid_astar_pose_path_hash": row.get("hybrid_astar_pose_path_hash"),
            "hybrid_astar_trajectory_kind": row.get("hybrid_astar_trajectory_kind"),
            "legacy_grid_astar_path_cost": row.get("legacy_grid_astar_path_cost"),
            "hybrid_vs_grid_path_cost_delta": row.get("hybrid_vs_grid_path_cost_delta"),
            "default_astar_replaced": row.get("default_astar_replaced"),
            "hybrid_astar_ackermann_feasible_claimed": row.get("hybrid_astar_ackermann_feasible_claimed"),
            "final_coverage_rate_after_step": 0.1,
            "path_cost": row.get("legacy_grid_astar_path_cost"),
            "soft_risk_exposure": 0.0,
            "hard_risk_violation": False,
        },
    }
    if mutate:
        payload["info"].update(mutate)
    return payload


def _batch_like(reward_rows: list[dict[str, Any]], transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batch_rows = []
    for reward_row, transition in zip(reward_rows, transitions):
        batch_like = dict(reward_row)
        batch_like["info"] = transition.get("info")
        batch_like["action_index"] = transition.get("action_index")
        batch_like["reward_metrics"] = reward_row.get("metrics")
        batch_rows.append(batch_like)
    return batch_rows


def _mutated_reward_row(row: dict[str, Any], **updates: Any) -> dict[str, Any]:
    payload = dict(row)
    if "contract" in updates:
        payload["hybrid_astar_path_cost_reward_contract"] = bool(updates.pop("contract"))
    if "fallback" in updates:
        payload["point_grid_path_cost_fallback_used"] = bool(updates.pop("fallback"))
    payload.update(updates)
    return payload


def _route(
    boundary_reasons: list[str],
    input_reasons: list[str],
    contract: dict[str, Any],
    compatibility: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage24_2_boundary_rejected"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage24_2_inputs_missing_or_untrusted"
    if contract["hybrid_astar_path_cost_contract_missing_count"] or contract["point_grid_path_cost_fallback_used_count"]:
        return "failed", ROUTE_SOURCE, "hybrid_astar_path_cost_provenance_missing"
    if contract["hybrid_grid_cost_material_row_count"] and contract["hybrid_grid_cost_diff_but_reward_equal_count"]:
        return "failed", ROUTE_DISCRIMINATION, "hybrid_grid_cost_not_discriminated"
    batch_gate = compatibility["stage21_3_batch_gate_audit"]
    if not compatibility["stage21_2_hybrid_path_cost_supported"]:
        return "failed", ROUTE_SOURCE, "stage21_2_hybrid_path_cost_not_supported"
    if not (
        batch_gate["accepts_hybrid_path_cost_batch"]
        and batch_gate["rejects_grid_only_path_cost_batch"]
        and batch_gate["rejects_ackermann_claim_batch"]
    ):
        return "failed", ROUTE_BATCH_GATE, "stage21_3_hybrid_path_cost_gate_missing"
    return "passed", ROUTE_STAGE24_3, "hybrid_astar_reward_path_cost_contract_ready"


def _route_blocking_reasons(route: str, contract: dict[str, Any], compatibility: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if route == ROUTE_SOURCE:
        if contract.get("hybrid_astar_path_cost_contract_missing_count"):
            reasons.append("hybrid_astar_path_cost_contract_missing")
        if contract.get("point_grid_path_cost_fallback_used_count"):
            reasons.append("point_grid_path_cost_fallback_used")
        if not compatibility.get("stage21_2_hybrid_path_cost_supported"):
            reasons.append("stage21_2_hybrid_path_cost_not_supported")
    if route == ROUTE_DISCRIMINATION:
        reasons.append("hybrid_grid_cost_diff_but_reward_equal")
    if route == ROUTE_BATCH_GATE:
        reasons.append("stage21_3_hybrid_path_cost_gate_missing")
    return reasons


def _input_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    root = Path(config["stage24_1_root"])
    summary_path = root / STAGE24_1_SUMMARY_FILE
    recommendation_path = root / STAGE24_1_RECOMMENDATION_FILE
    audit_path = Path(config["candidate_audit_path"])
    if not summary_path.is_file():
        return ["missing_stage24_1_summary"]
    summary = _read_json(summary_path)
    if summary.get("status") != "passed":
        reasons.append("stage24_1_not_passed")
    if summary.get("next_required_change") != "run_stage24_2_hybrid_astar_reward_path_cost_contract":
        reasons.append("stage24_1_route_not_stage24_2")
    if summary.get("path_cost_source_recommendation") != HYBRID_ASTAR_PATH_COST_SOURCE:
        reasons.append("stage24_1_path_cost_source_not_hybrid_astar")
    if summary.get("default_astar_replaced") is not False:
        reasons.append("stage24_1_default_astar_replaced")
    if summary.get("ackermann_feasible_claimed") is not False:
        reasons.append("stage24_1_ackermann_feasible_claimed")
    if not audit_path.is_file():
        reasons.append("missing_stage24_1_candidate_hybrid_path_cost_audit")
    if recommendation_path.is_file():
        recommendation = _read_json(recommendation_path)
        recommended = recommendation.get("recommended_path_cost_source") or recommendation.get("path_cost_source_recommendation")
        if recommended != HYBRID_ASTAR_PATH_COST_SOURCE:
            reasons.append("stage24_1_recommendation_not_hybrid_astar")
        if recommendation.get("default_astar_replaced") is not False:
            reasons.append("stage24_1_recommendation_default_astar_replaced")
        if recommendation.get("ackermann_feasible_claimed") is not False:
            reasons.append("stage24_1_recommendation_ackermann_feasible_claimed")
    else:
        reasons.append("missing_stage24_1_path_cost_source_recommendation")
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
        "stage24_1_root": DEFAULT_STAGE24_1_ROOT,
        "coverage_first_reward_profile": "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json",
        "coverage_denominator_cells": 100.0,
        "replay_obstacle_aware_new_visible_cell_count": 4.0,
        "slope_blocked_source_kind": "slope_blocked_as_obstacle_proxy",
        "max_replay_rows": 5000,
        "stage24_2_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    config["stage24_1_root"] = str(_resolve_path(Path(str(config["stage24_1_root"])), repo_root))
    config["coverage_first_reward_profile"] = str(_resolve_path(Path(str(config["coverage_first_reward_profile"])), repo_root))
    config["coverage_denominator_cells"] = _positive_float(config["coverage_denominator_cells"], "coverage_denominator_cells")
    config["replay_obstacle_aware_new_visible_cell_count"] = _nonnegative_float(
        config["replay_obstacle_aware_new_visible_cell_count"],
        "replay_obstacle_aware_new_visible_cell_count",
    )
    config["max_replay_rows"] = _positive_int(config["max_replay_rows"], "max_replay_rows")
    config["slope_blocked_source_kind"] = str(config.get("slope_blocked_source_kind") or "")
    if config["slope_blocked_source_kind"] != "slope_blocked_as_obstacle_proxy":
        raise ValueError("slope_blocked_source_kind must be slope_blocked_as_obstacle_proxy")
    if "candidate_audit_path" not in config:
        config["candidate_audit_path"] = str(Path(config["stage24_1_root"]) / STAGE24_1_AUDIT_FILE)
    else:
        config["candidate_audit_path"] = str(_resolve_path(Path(str(config["candidate_audit_path"])), repo_root))
    return config


def _row_has_hybrid_path_source(row: dict[str, Any]) -> bool:
    return (
        row.get("path_cost_source_recommendation") == HYBRID_ASTAR_PATH_COST_SOURCE
        and row.get("hybrid_astar_trajectory_kind") == "hybrid_astar_pose_path"
        and _finite(row.get("hybrid_astar_path_cost")) is not None
        and _finite(row.get("hybrid_astar_path_cost")) > 0.0
        and isinstance(row.get("hybrid_astar_pose_path_hash"), str)
        and bool(str(row.get("hybrid_astar_pose_path_hash")).strip())
        and _finite(row.get("legacy_grid_astar_path_cost")) is not None
        and _finite(row.get("hybrid_vs_grid_path_cost_delta")) is not None
        and row.get("default_astar_replaced") is False
        and row.get("hybrid_astar_ackermann_feasible_claimed") is False
    )


def _row_has_hybrid_reward_contract(row: dict[str, Any]) -> bool:
    if not (
        row.get("hybrid_astar_path_cost_reward_contract") is True
        and row.get("path_cost_source") == HYBRID_ASTAR_PATH_COST_SOURCE
        and _finite(row.get("hybrid_astar_path_cost")) is not None
        and _finite(row.get("hybrid_astar_path_cost")) > 0.0
        and isinstance(row.get("hybrid_astar_pose_path_hash"), str)
        and bool(str(row.get("hybrid_astar_pose_path_hash")).strip())
        and row.get("hybrid_astar_trajectory_kind") == "hybrid_astar_pose_path"
        and _finite(row.get("legacy_grid_astar_path_cost")) is not None
        and _finite(row.get("hybrid_vs_grid_path_cost_delta")) is not None
        and row.get("default_astar_replaced") is False
        and row.get("hybrid_astar_ackermann_feasible_claimed") is False
        and row.get("point_grid_path_cost_fallback_used") is False
    ):
        return False
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return _finite(metrics.get("path_cost_m")) == _finite(row.get("hybrid_astar_path_cost"))


def _reward_or_components_different(left: Any, right: Any) -> bool:
    if _different(left.reward, right.reward):
        return True
    left_components = left.components if isinstance(left.components, dict) else {}
    right_components = right.components if isinstance(right.components, dict) else {}
    keys = set(left_components) | set(right_components)
    return any(_different(left_components.get(key), right_components.get(key)) for key in keys)


def _coverage_per_cost(coverage_count: float | None, path_cost: float | None) -> float | None:
    if coverage_count is None or path_cost is None or path_cost <= 0.0:
        return None
    return float(coverage_count) / float(path_cost)


def _different(left: Any, right: Any) -> bool:
    left_value = _finite(left)
    right_value = _finite(right)
    if left_value is None or right_value is None:
        return False
    return abs(left_value - right_value) > 1.0e-12


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _positive_float(value: Any, name: str) -> float:
    number = _finite(value)
    if number is None or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return float(number)


def _nonnegative_float(value: Any, name: str) -> float:
    number = _finite(value)
    if number is None or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return float(number)


def _positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number <= 0:
        raise ValueError(f"{name} must be > 0")
    return number


def _unique(values: list[str]) -> list[str]:
    return sorted(set(values))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage24.2 Hybrid A* Reward Path Cost Contract",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- coverage_source: `{summary['coverage_source']}`",
            f"- path_cost_source: `{summary['path_cost_source']}`",
            f"- reward_replay_row_count: `{summary['reward_replay_row_count']}`",
            "- hybrid_astar_path_cost_contract_missing_count: "
            f"`{summary['hybrid_astar_path_cost_contract_missing_count']}`",
            f"- hybrid_grid_cost_material_row_count: `{summary['hybrid_grid_cost_material_row_count']}`",
            f"- default_astar_replaced: `{summary['default_astar_replaced']}`",
            f"- ackermann_feasible_claimed: `{summary['ackermann_feasible_claimed']}`",
            "",
            "Stage24.2 only validates reward and batch contracts. It does not run PPO, publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
        ]
    ) + "\n"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
