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
    coverage_first_profile_hash,
    load_coverage_first_reward_profile,
)


CONFIG_SCHEMA_VERSION = "xunce-stage21-2-coverage-first-ppo-reward-contract-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-2-coverage-first-reward-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-2-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-2-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_2_coverage_first_ppo_reward_contract_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/"
    "outputs/path_feedback_batch_xunce_stage21_2_coverage_first_ppo_reward_contract_v1"
)

SUMMARY_FILE = "xunce-stage21-2-coverage-first-reward-summary.json"
EVALUATION_FILE = "xunce-stage21-2-reward-contract-evaluation.jsonl"
PROFILE_AUDIT_FILE = "xunce-stage21-2-profile-audit.json"
ROUTING_FILE = "xunce-stage21-2-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-2-report.md"
MANIFEST_FILE = "xunce-stage21-2-manifest.json"

ROUTE_BOUNDARY = "resolve_stage21_2_coverage_first_reward_boundary_rejections"
ROUTE_INPUTS = "rerun_stage21_2_required_inputs"
ROUTE_REPAIR = "repair_stage21_2_coverage_first_reward_contract"
ROUTE_HORIZON = "stage21_12_rollout_horizon_or_mission_budget_scaling_for_99pct_coverage"
ROUTE_STAGE21_3 = "implement_stage21_3_ppo_batch_validation"
THETA_COVERAGE_SOURCE = "theta_aware_sensor_footprint/v1"

BOUNDARY_FIELDS = (
    "stage21_2_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.2 coverage-first PPO reward contract audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    try:
        summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
            config_path=Path(args.config),
            output_root=Path(args.output_root),
            repo_root=Path(args.repo_root),
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "reward_evaluation_row_count": summary["reward_evaluation_row_count"],
                "best_final_coverage_rate": summary["best_final_coverage_rate"],
                "stage21_2_authorized": summary["stage21_2_authorized"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage21_2_coverage_first_ppo_reward_contract(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root=repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    profile = load_coverage_first_reward_profile(config["coverage_first_reward_profile"])

    boundary_reasons = _boundary_rejections(config)
    input_reasons = _input_rejections(config, profile)
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    stage21_1_summary: dict[str, Any] = {}
    if not boundary_reasons and not input_reasons:
        stage21_1_root = Path(config["stage21_1_collector_root"])
        stage21_1_summary = _read_json(stage21_1_root / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json")
        transitions = _read_jsonl(stage21_1_root / "xunce-stage21-1-ppo-trainable-batch.jsonl")
        episodes = _read_jsonl(stage21_1_root / "xunce-stage21-1-ppo-rollout-episodes.jsonl")
        rows = _evaluate_transitions(transitions, profile=profile, config=config)

    profile_audit = _profile_audit(profile, rows)
    blocking = list(boundary_reasons + input_reasons)
    blocking.extend(_contract_rejections(rows, profile_audit))
    best_final = _best_final_coverage(episodes)
    route = ROUTE_STAGE21_3
    status = "passed"
    if boundary_reasons:
        route = ROUTE_BOUNDARY
        status = "failed"
    elif input_reasons:
        route = ROUTE_INPUTS
        status = "failed"
    elif blocking:
        route = ROUTE_REPAIR
        status = "failed"
    elif episodes and all(_episode_steps(row) >= profile.horizon_steps for row in episodes) and best_final < profile.target_final_coverage_rate:
        route = profile.mission_budget_route_when_below_target
        status = "failed"

    return _write_outputs(
        config=config,
        profile=profile,
        output_root=output_root,
        config_path=_resolve_path(config_path, repo_root),
        stage21_1_summary=stage21_1_summary,
        rows=rows,
        episodes=episodes,
        profile_audit=profile_audit,
        blocking_reason_codes=blocking,
        route=route,
        status=status,
    )


def _evaluate_transitions(transitions: list[dict[str, Any]], *, profile: Any, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config = config or {}
    require_theta_reward = bool(config.get("require_theta_aware_reward_contract", False))
    theta_denominator = _positive_or_default(config.get("theta_coverage_denominator_cells"), 1.0)
    rows: list[dict[str, Any]] = []
    for row in transitions:
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        theta_reward = _theta_reward_provenance(row, theta_denominator=theta_denominator)
        if require_theta_reward and theta_reward["theta_aware_reward_contract"]:
            coverage_rate_delta = theta_reward["coverage_rate_delta"]
            coverage_per_cost = theta_reward["theta_coverage_gain_per_path_cost"]
            coverage_source = THETA_COVERAGE_SOURCE
            point_only_fallback_used = False
        else:
            coverage_rate_delta = info.get("coverage_rate_delta")
            coverage_per_cost = info.get("coverage_per_cost")
            coverage_source = info.get("coverage_source") or "point_or_path_line_coverage_legacy"
            point_only_fallback_used = bool(require_theta_reward)
        metrics = {
            "coverage_rate_delta": coverage_rate_delta,
            "coverage_per_cost": coverage_per_cost,
            "coverage_progress_rate": info.get("final_coverage_rate_after_step"),
            "final_coverage_rate": info.get("final_coverage_rate_after_step"),
            "path_cost_m": info.get("path_cost"),
            "soft_risk_exposure": info.get("soft_risk_exposure"),
            "done": row.get("done"),
            "hard_risk_flags": [] if not info.get("hard_risk_violation") else ["hard_risk_violation"],
            "hard_risk_violation_count": 1 if info.get("hard_risk_violation") else 0,
            "open_grid_fallback_used": False,
        }
        result = compute_coverage_first_reward_components(metrics, profile)
        rows.append(
            {
                "schema_version": "xunce-stage21-2-reward-contract-evaluation/v1",
                "transition_id": row.get("transition_id"),
                "scenario_id": row.get("scenario_id"),
                "step_index": row.get("step_index"),
                "done": bool(row.get("done")),
                "metrics": metrics,
                "reward": result.reward,
                "components": result.components,
                "trainable": result.trainable and bool(row.get("trainable", False)),
                "reason_codes": result.reason_codes,
                "theta_aware_reward_contract": bool(require_theta_reward and theta_reward["theta_aware_reward_contract"]),
                "coverage_source": coverage_source,
                "candidate_viewpoint": theta_reward["candidate_viewpoint"],
                "candidate_theta_deg": theta_reward["candidate_theta_deg"],
                "theta_new_visible_cell_count": theta_reward["theta_new_visible_cell_count"],
                "theta_coverage_hash": theta_reward["theta_coverage_hash"],
                "theta_coverage_gain_per_path_cost": theta_reward["theta_coverage_gain_per_path_cost"],
                "theta_coverage_denominator_cells": theta_denominator,
                "point_only_reward_fallback_used": point_only_fallback_used,
                "risk_deduplication_applied": result.risk_deduplication_applied,
                "profile_id": result.profile_id,
                "profile_version": result.profile_version,
                "profile_hash": result.profile_hash,
            }
        )
    return rows


def _profile_audit(profile: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    component_sets = {tuple(sorted((row.get("components") or {}).keys())) for row in rows}
    reward_values = [_finite(row.get("reward")) for row in rows]
    return {
        "schema_version": "xunce-stage21-2-profile-audit/v1",
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "profile_hash_recomputed": coverage_first_profile_hash(profile),
        "target_final_coverage_rate": profile.target_final_coverage_rate,
        "horizon_steps": profile.horizon_steps,
        "component_keys": list(profile.component_keys),
        "observed_component_set_count": len(component_sets),
        "all_rewards_finite": all(value is not None for value in reward_values),
        "all_profile_hashes_match": all(row.get("profile_hash") == profile.profile_hash for row in rows),
        "hard_risk_positive_reward_count": sum(
            1
            for row in rows
            if "hard_risk_rejected" in (row.get("reason_codes") or []) and _finite(row.get("reward")) is not None and float(row["reward"]) > 0.0
        ),
        "soft_risk_deduplication_applied_count": sum(1 for row in rows if row.get("risk_deduplication_applied") is True),
        "theta_aware_reward_contract_count": sum(1 for row in rows if row.get("theta_aware_reward_contract") is True),
        "theta_reward_contract_missing_count": _theta_reward_contract_missing_count(rows),
        "point_only_reward_fallback_used_count": sum(1 for row in rows if row.get("point_only_reward_fallback_used") is True),
    }


def _contract_rejections(rows: list[dict[str, Any]], profile_audit: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not rows:
        reasons.append("missing_stage21_1_trainable_batch_rows")
    if profile_audit["profile_hash"] != profile_audit["profile_hash_recomputed"]:
        reasons.append("coverage_first_profile_hash_unstable")
    if profile_audit["observed_component_set_count"] not in {0, 1}:
        reasons.append("coverage_first_component_set_inconsistent")
    if not profile_audit["all_rewards_finite"]:
        reasons.append("coverage_first_reward_non_finite")
    if not profile_audit["all_profile_hashes_match"]:
        reasons.append("coverage_first_profile_hash_mismatch")
    if profile_audit["hard_risk_positive_reward_count"] > 0:
        reasons.append("hard_risk_positive_reward_detected")
    if profile_audit.get("point_only_reward_fallback_used_count", 0) > 0:
        reasons.append("theta_aware_reward_contract_missing")
    return reasons


def _write_outputs(
    *,
    config: dict[str, Any],
    profile: Any,
    output_root: Path,
    config_path: Path,
    stage21_1_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    profile_audit: dict[str, Any],
    blocking_reason_codes: list[str],
    route: str,
    status: str,
) -> dict[str, Any]:
    generated_at = _utc_now()
    best_final = _best_final_coverage(episodes)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "primary_route": route,
        "stage21_2_authorized": False,
        "stage21_3_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "next_required_change": route,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "target_final_coverage_rate": profile.target_final_coverage_rate,
        "horizon_steps": profile.horizon_steps,
        "stage21_1_collector_root": config["stage21_1_collector_root"],
        "stage21_1_status": stage21_1_summary.get("status"),
        "stage21_1_profile_hash": stage21_1_summary.get("profile_hash"),
        "reward_evaluation_row_count": len(rows),
        "episode_count": len(episodes),
        "best_final_coverage_rate": best_final,
        "all_rewards_finite": profile_audit["all_rewards_finite"],
        "all_profile_hashes_match": profile_audit["all_profile_hashes_match"],
        "hard_risk_positive_reward_count": profile_audit["hard_risk_positive_reward_count"],
        "soft_risk_deduplication_applied_count": profile_audit["soft_risk_deduplication_applied_count"],
        "theta_aware_reward_contract_required": bool(config.get("require_theta_aware_reward_contract", False)),
        "theta_aware_reward_contract_count": profile_audit["theta_aware_reward_contract_count"],
        "theta_reward_contract_missing_count": profile_audit["theta_reward_contract_missing_count"],
        "point_only_reward_fallback_used_count": profile_audit["point_only_reward_fallback_used_count"],
        "blocking_reason_codes": _unique(blocking_reason_codes),
        "reason_codes": _unique(blocking_reason_codes),
        "stage21_2_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str((output_root / SUMMARY_FILE).resolve()),
        "evaluation": str((output_root / EVALUATION_FILE).resolve()),
        "profile_audit": str((output_root / PROFILE_AUDIT_FILE).resolve()),
        "routing": str((output_root / ROUTING_FILE).resolve()),
        "report": str((output_root / REPORT_FILE).resolve()),
        "manifest": str((output_root / MANIFEST_FILE).resolve()),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path.resolve()),
        "artifacts": {
            "summary": summary["summary"],
            "evaluation": summary["evaluation"],
            "profile_audit": summary["profile_audit"],
            "routing": summary["routing"],
            "report": summary["report"],
        },
        "summary_status": status,
        "next_required_change": route,
    }
    _write_jsonl(output_root / EVALUATION_FILE, rows)
    _write_json(output_root / PROFILE_AUDIT_FILE, profile_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _load_config(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    for key in ("stage21_1_collector_root", "coverage_first_reward_profile"):
        if not isinstance(config.get(key), str) or not str(config[key]).strip():
            raise ConfigError(f"{key} must be a non-empty path string")
        config[key] = str(_resolve_path(Path(config[key]), repo_root))
    config["canary_traffic_fraction"] = _nonnegative_float(config.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    config["require_theta_aware_reward_contract"] = bool(config.get("require_theta_aware_reward_contract", False))
    config["theta_coverage_denominator_cells"] = _positive_or_default(config.get("theta_coverage_denominator_cells"), 1.0)
    for field in BOUNDARY_FIELDS:
        config.setdefault(field, False)
    return config


def _input_rejections(config: dict[str, Any], profile: Any) -> list[str]:
    reasons: list[str] = []
    root = Path(config["stage21_1_collector_root"])
    summary_path = root / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json"
    batch_path = root / "xunce-stage21-1-ppo-trainable-batch.jsonl"
    episodes_path = root / "xunce-stage21-1-ppo-rollout-episodes.jsonl"
    if not summary_path.is_file():
        reasons.append("missing_stage21_1_collector_summary")
    else:
        summary = _read_json(summary_path)
        if summary.get("status") != "passed":
            reasons.append("stage21_1_collector_not_passed")
        if summary.get("next_required_change") != "implement_stage21_2_coverage_first_ppo_reward_contract":
            reasons.append("stage21_1_route_not_stage21_2_reward_contract")
    if not batch_path.is_file():
        reasons.append("missing_stage21_1_trainable_batch")
    if not episodes_path.is_file():
        reasons.append("missing_stage21_1_episodes")
    if profile.profile_version not in {"stage21-coverage-first-v1", "stage21-coverage-constrained-v2"}:
        reasons.append("stage21_2_requires_supported_coverage_reward_profile")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in BOUNDARY_FIELDS:
        if config.get(field) is not False:
            reasons.append(field)
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _best_final_coverage(episodes: list[dict[str, Any]]) -> float:
    values = [_finite(row.get("final_coverage_rate")) for row in episodes]
    values = [float(value) for value in values if value is not None]
    return max(values) if values else 0.0


def _episode_steps(row: dict[str, Any]) -> int:
    try:
        return int(row.get("rollout_steps", 0))
    except (TypeError, ValueError):
        return 0


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 21.2 Coverage-First PPO Reward Contract",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- reward_evaluation_row_count: `{summary['reward_evaluation_row_count']}`",
            f"- target_final_coverage_rate: `{summary['target_final_coverage_rate']}`",
            f"- best_final_coverage_rate: `{summary['best_final_coverage_rate']}`",
            f"- hard_risk_positive_reward_count: `{summary['hard_risk_positive_reward_count']}`",
            f"- theta_aware_reward_contract_required: `{summary.get('theta_aware_reward_contract_required', False)}`",
            f"- theta_reward_contract_missing_count: `{summary.get('theta_reward_contract_missing_count', 0)}`",
            "",
            "Stage 21.2 validates the reward contract only. It does not run PPO, publish checkpoints, replace default policy, connect a real executor, or start canary traffic.",
        ]
    ) + "\n"


def _theta_reward_contract_missing_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("theta_aware_reward_contract") is not True)


def _theta_reward_provenance(row: dict[str, Any], *, theta_denominator: float) -> dict[str, Any]:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    action_index = _int_or_none(row.get("action_index"))
    viewpoint = _selected_value(info, "candidate_viewpoint", "candidate_viewpoints", action_index)
    theta_deg = _selected_value(info, "candidate_theta_deg", "candidate_theta_degs", action_index)
    theta_new = _selected_value(
        info,
        "theta_new_visible_cell_count",
        "theta_new_visible_cell_counts",
        action_index,
    )
    theta_hash = _selected_value(info, "theta_coverage_hash", "theta_coverage_hashes", action_index)
    theta_cpc = _selected_value(
        info,
        "theta_coverage_gain_per_path_cost",
        "theta_coverage_gain_per_path_costs",
        action_index,
    )
    if viewpoint is None:
        viewpoint = row.get("candidate_viewpoint")
    if theta_deg is None:
        theta_deg = row.get("candidate_theta_deg")
    if theta_new is None:
        theta_new = row.get("theta_new_visible_cell_count")
    if theta_hash is None:
        theta_hash = row.get("theta_coverage_hash")
    if theta_cpc is None:
        theta_cpc = row.get("theta_coverage_gain_per_path_cost")
    theta_count = _finite(theta_new)
    if theta_deg is None and isinstance(viewpoint, list) and len(viewpoint) >= 3:
        theta_deg = viewpoint[2]
    contract = (
        _present_viewpoint(viewpoint)
        and _finite(theta_deg) is not None
        and theta_count is not None
        and isinstance(theta_hash, str)
        and bool(theta_hash.strip())
    )
    return {
        "theta_aware_reward_contract": bool(contract),
        "candidate_viewpoint": viewpoint,
        "candidate_theta_deg": int(float(theta_deg)) if _finite(theta_deg) is not None else None,
        "theta_new_visible_cell_count": int(theta_count) if theta_count is not None else None,
        "theta_coverage_hash": theta_hash,
        "theta_coverage_gain_per_path_cost": _finite(theta_cpc),
        "coverage_rate_delta": (float(theta_count) / theta_denominator) if theta_count is not None else None,
    }


def _selected_value(info: dict[str, Any], singular_key: str, plural_key: str, action_index: int | None) -> Any:
    if singular_key in info:
        value = info.get(singular_key)
        if action_index is not None and isinstance(value, list):
            if singular_key == "candidate_viewpoint":
                if value and isinstance(value[0], list) and 0 <= action_index < len(value):
                    return value[action_index]
                return value
            if 0 <= action_index < len(value):
                return value[action_index]
        return value
    values = info.get(plural_key)
    if action_index is not None and isinstance(values, list) and 0 <= action_index < len(values):
        return values[action_index]
    return None


def _present_viewpoint(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 3


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"JSON file is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"JSON root must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ConfigError(f"JSONL file does not exist: {path}") from exc
    for line in lines:
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _nonnegative_float(value: Any, field: str) -> float:
    parsed = _finite(value)
    if parsed is None or parsed < 0.0:
        raise ConfigError(f"{field} must be finite and >= 0")
    return parsed


def _positive_or_default(value: Any, default: float) -> float:
    parsed = _finite(value)
    if parsed is None:
        return float(default)
    if parsed <= 0.0:
        raise ConfigError("theta_coverage_denominator_cells must be finite and > 0")
    return float(parsed)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
