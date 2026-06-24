from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by script execution
    from model_explorer.policy.coverage_first_reward import (
        compute_coverage_first_reward_components,
        load_coverage_first_reward_profile,
    )
except ModuleNotFoundError:  # pragma: no cover
    import sys

    repo_root_for_import = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root_for_import / "model-explorer" / "src"))
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


CONFIG_SCHEMA_VERSION = "xunce-stage22-2-theta-aware-coverage-reward-contract-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage22-2-summary/v1"
REPLAY_ROW_SCHEMA_VERSION = "xunce-stage22-2-reward-replay-row/v1"
CONTRACT_AUDIT_SCHEMA_VERSION = "xunce-stage22-2-theta-reward-contract-audit/v1"
COMPAT_AUDIT_SCHEMA_VERSION = "xunce-stage22-2-stage21-2-compatibility-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage22-2-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage22-2-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage22_2_theta_aware_coverage_reward_contract_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage22_theta_aware_sensor_action_space/"
    "outputs/path_feedback_batch_xunce_stage22_2_theta_aware_coverage_reward_contract_v1"
)

SUMMARY_FILE = "xunce-stage22-2-summary.json"
CONTRACT_AUDIT_FILE = "xunce-stage22-2-theta-reward-contract-audit.json"
REWARD_REPLAY_FILE = "xunce-stage22-2-reward-replay.jsonl"
COMPAT_AUDIT_FILE = "xunce-stage22-2-stage21-2-compatibility-audit.json"
ROUTING_FILE = "xunce-stage22-2-next-stage-routing.json"
REPORT_FILE = "xunce-stage22-2-report.md"
MANIFEST_FILE = "xunce-stage22-2-manifest.json"

ROUTE_INPUTS = "rerun_stage22_2_required_inputs"
ROUTE_PROVENANCE = "repair_stage22_2_theta_reward_provenance"
ROUTE_DISCRIMINATION = "repair_stage22_2_viewpoint_reward_discrimination"
ROUTE_BATCH_REWARD = "repair_stage22_2_stage21_batch_reward_contract"
ROUTE_STAGE22_3 = "run_stage22_3_theta_aware_ppo_collector_smoke"
ROUTE_BOUNDARY = "resolve_stage22_2_boundary_rejections"

THETA_COVERAGE_SOURCE = "theta_aware_sensor_footprint/v1"

BOUNDARY_FIELDS = (
    "stage22_2_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage22.2 theta-aware coverage reward contract audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage22_2_theta_aware_coverage_reward_contract(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage22_2_theta_aware_coverage_reward_contract(
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
    profile = load_coverage_first_reward_profile(config["coverage_first_reward_profile"])
    stage22_1_summary: dict[str, Any] = {}
    viewpoint_rows: list[dict[str, Any]] = []
    if not input_reasons:
        stage22_1_root = Path(config["stage22_1_root"])
        stage22_1_summary = _read_json(stage22_1_root / "xunce-stage22-1-summary.json")
        viewpoint_rows = _read_jsonl(stage22_1_root / "xunce-stage22-1-viewpoint-candidate-audit.jsonl")

    replay_rows = _replay_rewards(viewpoint_rows, profile=profile, config=config)
    contract_audit = _contract_audit(replay_rows)
    compatibility_audit = _compatibility_audit(replay_rows, profile=profile, config=config)
    status, route, route_reason = _route(boundary_reasons, input_reasons, contract_audit, compatibility_audit)
    blocking = _unique(boundary_reasons + input_reasons + _route_blocking_reasons(route, contract_audit, compatibility_audit))

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": "xunce-stage22-2-theta-aware-coverage-reward-contract",
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking,
        "stage22_1_root": config["stage22_1_root"],
        "stage22_1_status": stage22_1_summary.get("status"),
        "stage22_1_next_required_change": stage22_1_summary.get("next_required_change"),
        "coverage_first_reward_profile": config["coverage_first_reward_profile"],
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "coverage_denominator_cells": float(config["coverage_denominator_cells"]),
        "viewpoint_reward_replay_row_count": len(replay_rows),
        **{key: contract_audit[key] for key in (
            "theta_aware_reward_contract_count",
            "point_only_reward_fallback_used_count",
            "theta_reward_contract_missing_count",
            "coverage_diff_group_count",
            "reward_discriminated_coverage_diff_group_count",
            "coverage_diff_but_reward_equal_group_count",
        )},
        "stage21_2_theta_reward_supported": compatibility_audit["stage21_2_theta_reward_supported"],
        "stage21_3_rejects_point_only_reward_batch": compatibility_audit["stage21_3_rejects_point_only_reward_batch"],
        "stage21_3_accepts_theta_reward_batch": compatibility_audit["stage21_3_accepts_theta_reward_batch"],
        "stage21_3_rejects_mismatched_theta_reward_batch": compatibility_audit["stage21_3_rejects_mismatched_theta_reward_batch"],
        "release_or_training_authorized": False,
        "stage22_2_authorized": False,
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
        "stage22_2_authorized": False,
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
        "theta_reward_contract_audit": str(output_root / CONTRACT_AUDIT_FILE),
        "reward_replay": str(output_root / REWARD_REPLAY_FILE),
        "stage21_2_compatibility_audit": str(output_root / COMPAT_AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / CONTRACT_AUDIT_FILE, contract_audit)
    _write_jsonl(output_root / REWARD_REPLAY_FILE, replay_rows)
    _write_json(output_root / COMPAT_AUDIT_FILE, compatibility_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    _write_json(output_root / MANIFEST_FILE, manifest)
    return summary


def _replay_rewards(rows: list[dict[str, Any]], *, profile: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    replay_rows: list[dict[str, Any]] = []
    denominator = float(config["coverage_denominator_cells"])
    for row in rows:
        theta_new = _finite(row.get("theta_new_visible_cell_count"))
        path_cost = _path_cost_from_row(row)
        coverage_rate_delta = (float(theta_new) / denominator) if theta_new is not None else None
        metrics = {
            "coverage_rate_delta": coverage_rate_delta,
            "coverage_per_cost": row.get("theta_coverage_gain_per_path_cost"),
            "coverage_progress_rate": coverage_rate_delta,
            "final_coverage_rate": coverage_rate_delta,
            "path_cost_m": path_cost,
            "soft_risk_exposure": 0.0,
            "done": False,
            "hard_risk_flags": [] if row.get("action_mask_valid") is not False else ["invalid_viewpoint_action"],
            "hard_risk_violation_count": 0 if row.get("action_mask_valid") is not False else 1,
            "open_grid_fallback_used": False,
        }
        result = compute_coverage_first_reward_components(metrics, profile)
        replay_rows.append(
            {
                "schema_version": REPLAY_ROW_SCHEMA_VERSION,
                "scenario_id": row.get("scenario_id"),
                "step_index": row.get("step_index"),
                "candidate_cell": row.get("candidate_cell"),
                "candidate_viewpoint": row.get("candidate_viewpoint"),
                "candidate_theta_deg": row.get("candidate_theta_deg"),
                "viewpoint_index": row.get("viewpoint_index"),
                "base_candidate_index": row.get("base_candidate_index"),
                "candidate_set_hash": row.get("candidate_set_hash"),
                "sensor_model_id": row.get("sensor_model_id"),
                "sensor_fov_deg": row.get("sensor_fov_deg"),
                "sensor_range_cells": row.get("sensor_range_cells"),
                "theta_aware_reward_contract": True,
                "coverage_source": THETA_COVERAGE_SOURCE,
                "theta_new_visible_cell_count": row.get("theta_new_visible_cell_count"),
                "theta_coverage_hash": row.get("theta_coverage_hash"),
                "theta_coverage_gain_per_path_cost": row.get("theta_coverage_gain_per_path_cost"),
                "theta_coverage_denominator_cells": denominator,
                "point_only_reward_fallback_used": False,
                "metrics": metrics,
                "reward": result.reward,
                "components": result.components,
                "trainable": result.trainable,
                "reason_codes": result.reason_codes,
                "profile_id": result.profile_id,
                "profile_version": result.profile_version,
                "profile_hash": result.profile_hash,
            }
        )
    return replay_rows


def _contract_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("scenario_id")),
            int(row.get("step_index") or 0),
            json.dumps(row.get("candidate_cell"), sort_keys=True),
        )
        grouped.setdefault(key, []).append(row)
    coverage_diff_groups = 0
    reward_discriminated_groups = 0
    coverage_diff_reward_equal_groups = 0
    examples: list[dict[str, Any]] = []
    for key, group in grouped.items():
        coverage_values = {_finite(item.get("theta_new_visible_cell_count")) for item in group}
        coverage_values.discard(None)
        if len(coverage_values) <= 1:
            continue
        coverage_diff_groups += 1
        reward_values = {_rounded(item.get("reward")) for item in group}
        reward_values.discard(None)
        if len(reward_values) > 1:
            reward_discriminated_groups += 1
        else:
            coverage_diff_reward_equal_groups += 1
        if len(examples) < 5:
            examples.append(
                {
                    "scenario_id": key[0],
                    "step_index": key[1],
                    "candidate_cell": group[0].get("candidate_cell"),
                    "coverage_values": sorted(float(value) for value in coverage_values),
                    "reward_values": sorted(float(value) for value in reward_values),
                    "theta_values": [item.get("candidate_theta_deg") for item in group],
                }
            )
    return {
        "schema_version": CONTRACT_AUDIT_SCHEMA_VERSION,
        "reward_replay_row_count": len(rows),
        "theta_aware_reward_contract_count": sum(1 for row in rows if row.get("theta_aware_reward_contract") is True),
        "point_only_reward_fallback_used_count": sum(1 for row in rows if row.get("point_only_reward_fallback_used") is True),
        "theta_reward_contract_missing_count": sum(1 for row in rows if not _row_has_theta_reward_contract(row)),
        "coverage_source_theta_aware_count": sum(1 for row in rows if row.get("coverage_source") == THETA_COVERAGE_SOURCE),
        "coverage_diff_group_count": coverage_diff_groups,
        "reward_discriminated_coverage_diff_group_count": reward_discriminated_groups,
        "coverage_diff_but_reward_equal_group_count": coverage_diff_reward_equal_groups,
        "viewpoint_reward_discrimination_examples": examples,
    }


def _compatibility_audit(rows: list[dict[str, Any]], *, profile: Any, config: dict[str, Any]) -> dict[str, Any]:
    sample_rows = rows[:2]
    synthetic = []
    for index, row in enumerate(sample_rows):
        synthetic.append(
            {
                "transition_id": f"theta-sample:{index}",
                "scenario_id": row.get("scenario_id") or "s",
                "step_index": index,
                "done": False,
                "trainable": True,
                "action_index": 0,
                "info": {
                    "candidate_viewpoints": [row.get("candidate_viewpoint")],
                    "candidate_theta_deg": [row.get("candidate_theta_deg")],
                    "theta_new_visible_cell_counts": [row.get("theta_new_visible_cell_count")],
                    "theta_coverage_hashes": [row.get("theta_coverage_hash")],
                    "theta_coverage_gain_per_path_costs": [row.get("theta_coverage_gain_per_path_cost")],
                    "final_coverage_rate_after_step": 0.1,
                    "path_cost": _path_cost_from_row(row),
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
            "theta_coverage_denominator_cells": float(config["coverage_denominator_cells"]),
        },
    )
    stage21_2_supported = bool(stage21_2_rows) and all(_row_has_theta_reward_contract(row) for row in stage21_2_rows)
    point_only_missing_count = stage21_3._theta_reward_contract_missing_count([{"reward": 1.0}])
    theta_batch_rows = []
    for reward_row, transition in zip(stage21_2_rows, synthetic):
        batch_like = dict(reward_row)
        batch_like["info"] = transition.get("info")
        batch_like["action_index"] = transition.get("action_index")
        batch_like["reward_metrics"] = reward_row.get("metrics")
        theta_batch_rows.append(batch_like)
    theta_missing_count = stage21_3._theta_reward_contract_missing_count(theta_batch_rows)
    mismatch_rows = [dict(row) for row in theta_batch_rows[:1]]
    if mismatch_rows:
        mismatch_rows[0]["info"] = {
            "candidate_viewpoints": [[999, 999, 180]],
            "candidate_theta_deg": [180],
        }
        mismatch_rows[0]["action_index"] = 0
    mismatch_missing_count = stage21_3._theta_reward_contract_missing_count(mismatch_rows)
    return {
        "schema_version": COMPAT_AUDIT_SCHEMA_VERSION,
        "stage21_2_theta_reward_supported": stage21_2_supported,
        "stage21_2_theta_reward_row_count": len(stage21_2_rows),
        "stage21_2_point_only_reward_fallback_used_count": sum(1 for row in stage21_2_rows if row.get("point_only_reward_fallback_used") is True),
        "stage21_3_rejects_point_only_reward_batch": point_only_missing_count == 1,
        "stage21_3_accepts_theta_reward_batch": theta_missing_count == 0 and bool(stage21_2_rows),
        "stage21_3_rejects_mismatched_theta_reward_batch": mismatch_missing_count == len(mismatch_rows) and bool(mismatch_rows),
        "stage21_3_theta_reward_missing_count_for_theta_rows": theta_missing_count,
        "stage21_3_theta_reward_missing_count_for_mismatch_rows": mismatch_missing_count,
    }


def _route(
    boundary_reasons: list[str],
    input_reasons: list[str],
    contract: dict[str, Any],
    compatibility: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage22_2_boundary_rejected"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage22_2_inputs_missing_or_untrusted"
    if contract["theta_reward_contract_missing_count"] or contract["point_only_reward_fallback_used_count"]:
        return "failed", ROUTE_PROVENANCE, "theta_reward_provenance_missing_or_point_only_fallback_used"
    if contract["coverage_diff_but_reward_equal_group_count"]:
        return "failed", ROUTE_DISCRIMINATION, "viewpoint_coverage_diff_not_reflected_in_reward"
    if not compatibility["stage21_2_theta_reward_supported"]:
        return "failed", ROUTE_PROVENANCE, "stage21_2_theta_reward_support_missing"
    if (
        not compatibility["stage21_3_rejects_point_only_reward_batch"]
        or not compatibility["stage21_3_accepts_theta_reward_batch"]
        or not compatibility["stage21_3_rejects_mismatched_theta_reward_batch"]
    ):
        return "failed", ROUTE_BATCH_REWARD, "stage21_3_theta_reward_batch_gate_missing"
    return "passed", ROUTE_STAGE22_3, "theta_aware_reward_contract_ready_for_collector_smoke"


def _route_blocking_reasons(route: str, contract: dict[str, Any], compatibility: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if route == ROUTE_PROVENANCE:
        if contract.get("theta_reward_contract_missing_count"):
            reasons.append("theta_reward_contract_missing")
        if contract.get("point_only_reward_fallback_used_count"):
            reasons.append("point_only_reward_fallback_used")
        if not compatibility.get("stage21_2_theta_reward_supported"):
            reasons.append("stage21_2_theta_reward_not_supported")
    if route == ROUTE_DISCRIMINATION:
        reasons.append("theta_coverage_diff_reward_not_discriminated")
    if route == ROUTE_BATCH_REWARD:
        reasons.append("stage21_3_theta_reward_gate_missing")
    return reasons


def _input_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    root = Path(config["stage22_1_root"])
    summary_path = root / "xunce-stage22-1-summary.json"
    replay_path = root / "xunce-stage22-1-viewpoint-candidate-audit.jsonl"
    if not summary_path.is_file():
        reasons.append("missing_stage22_1_summary")
        return reasons
    summary = _read_json(summary_path)
    if summary.get("status") != "passed":
        reasons.append("stage22_1_not_passed")
    if summary.get("next_required_change") != "run_stage22_2_theta_aware_coverage_reward_contract":
        reasons.append("stage22_1_route_not_stage22_2")
    if summary.get("hash_missing_theta_count", 0) != 0:
        reasons.append("stage22_1_hash_missing_theta")
    if summary.get("mask_length_mismatch_count", 0) != 0:
        reasons.append("stage22_1_mask_length_mismatch")
    if not replay_path.is_file():
        reasons.append("missing_stage22_1_viewpoint_candidate_audit")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is not False]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _load_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(config_path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported config schema_version: {payload.get('schema_version')}")
    defaults = {
        "stage22_1_root": (
            "D:/CodexDownloads/lunar-path-planning/stage22_theta_aware_sensor_action_space/"
            "outputs/path_feedback_batch_xunce_stage22_1_theta_aware_candidate_viewpoint_generation_v1"
        ),
        "coverage_first_reward_profile": "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json",
        "coverage_denominator_cells": 1.0,
        "stage22_2_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    for key in ("stage22_1_root", "coverage_first_reward_profile"):
        config[key] = str(_resolve_path(Path(str(config[key])), repo_root))
    config["coverage_denominator_cells"] = _positive_float(config["coverage_denominator_cells"], "coverage_denominator_cells")
    return config


def _path_cost_from_row(row: dict[str, Any]) -> float:
    explicit = _finite(row.get("path_cost"))
    if explicit is not None and explicit > 0.0:
        return explicit
    theta_new = _finite(row.get("theta_new_visible_cell_count"))
    cpc = _finite(row.get("theta_coverage_gain_per_path_cost"))
    if theta_new is not None and cpc is not None and cpc > 1.0e-12:
        return max(1.0e-6, theta_new / cpc)
    return 1.0


def _row_has_theta_reward_contract(row: dict[str, Any]) -> bool:
    return (
        row.get("theta_aware_reward_contract") is True
        and row.get("coverage_source") == THETA_COVERAGE_SOURCE
        and isinstance(row.get("candidate_viewpoint"), list)
        and len(row.get("candidate_viewpoint")) >= 3
        and _finite(row.get("candidate_theta_deg")) is not None
        and _finite(row.get("theta_new_visible_cell_count")) is not None
        and _finite(row.get("theta_coverage_gain_per_path_cost")) is not None
        and _finite(row.get("theta_coverage_denominator_cells")) is not None
        and isinstance(row.get("theta_coverage_hash"), str)
        and bool(str(row.get("theta_coverage_hash")).strip())
        and row.get("point_only_reward_fallback_used") is False
    )


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage22.2 Theta-Aware Coverage Reward Contract",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- viewpoint_reward_replay_row_count: `{summary['viewpoint_reward_replay_row_count']}`",
            f"- theta_reward_contract_missing_count: `{summary['theta_reward_contract_missing_count']}`",
            f"- point_only_reward_fallback_used_count: `{summary['point_only_reward_fallback_used_count']}`",
            f"- coverage_diff_group_count: `{summary['coverage_diff_group_count']}`",
            f"- reward_discriminated_coverage_diff_group_count: `{summary['reward_discriminated_coverage_diff_group_count']}`",
            "",
            "Stage22.2 validates reward provenance only. It does not run PPO, publish checkpoints, replace default policy, connect executor, or start canary traffic.",
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


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rounded(value: Any) -> float | None:
    number = _finite(value)
    return round(number, 8) if number is not None else None


def _positive_float(value: Any, name: str) -> float:
    number = _finite(value)
    if number is None or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return float(number)


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
