from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
MODEL_EXPLORER_SRC = SCRIPT_DIR.parent / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now
from global_99_governance_common import global_99_boundary_defaults
from model_explorer.policy.canonical_reward import load_canonical_reward_profile


CONFIG_SCHEMA_VERSION = "xunce-stage18-9-trajectory-risk-boundary-reward-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage18-9-trajectory-risk-reward-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage18-9-next-stage-routing/v1"
SCENARIO_ROW_SCHEMA_VERSION = "xunce-stage18-9-scenario-trajectory-audit-row/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage18-9-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage18_9_trajectory_risk_boundary_reward_audit_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_stage18_9_trajectory_risk_boundary_reward_audit_v1"
DEFAULT_PROFILE = "configs/xunce_canonical_reward_guard_profile_v3.json"

COVERAGE_SUMMARY = "xunce-exploration-coverage-comparison-summary.json"
COVERAGE_AGGREGATE = "xunce-exploration-coverage-comparison-aggregate.json"
COVERAGE_PAIRS = "xunce-exploration-coverage-comparison-pairs.jsonl"
COVERAGE_EPISODES = "xunce-exploration-coverage-episodes.jsonl"
COVERAGE_MANIFEST = "xunce-exploration-coverage-comparison-manifest.json"
STAGE18_7_SUMMARY = "xunce-stage18-7-candidate-count-scaling-summary.json"

SUMMARY_FILE = "xunce-stage18-9-trajectory-risk-reward-summary.json"
SCENARIO_AUDIT_FILE = "xunce-stage18-9-scenario-trajectory-audit.jsonl"
BOUNDARY_FILE = "xunce-stage18-9-path-risk-boundary-summary.json"
REWARD_FILE = "xunce-stage18-9-reward-profile-v3-evaluation.json"
ROUTING_FILE = "xunce-stage18-9-next-stage-routing.json"
REPORT_FILE = "xunce-stage18-9-report.md"
MANIFEST_FILE = "xunce-stage18-9-manifest.json"

ROUTE_BOUNDARY = "resolve_stage18_9_boundary_rejections"
ROUTE_INPUTS = "rerun_required_stage18_9_inputs"
ROUTE_RISK = "repair_path_risk_boundary_filtering"
ROUTE_REWARD = "refine_coverage_cost_reward_weights"
ROUTE_SOFT_RISK = "calibrate_soft_risk_exposure_weight"
ROUTE_PREFLIGHT = "prepare_stage19_evaluator_critic_preflight"

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "default_policy_replacement_approved",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "runs_new_ppo_update",
    "real_world_release_approved",
    "real_world_performance_claimed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce Stage 18.9 trajectory risk/reward audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--coverage-comparison-root")
    parser.add_argument("--stage18-7-candidate-count-scaling-root")
    parser.add_argument("--canonical-profile")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_stage18_9_trajectory_risk_boundary_reward_audit(
            config_path=resolve_path(Path(args.config), repo_root).resolve(),
            output_root=resolve_path(Path(args.output_root), repo_root).resolve(),
            repo_root=repo_root,
            coverage_comparison_root=Path(args.coverage_comparison_root) if args.coverage_comparison_root else None,
            stage18_7_candidate_count_scaling_root=(
                Path(args.stage18_7_candidate_count_scaling_root)
                if args.stage18_7_candidate_count_scaling_root
                else None
            ),
            canonical_profile=Path(args.canonical_profile) if args.canonical_profile else None,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": summary["status"],
                "trajectory_guard_passed": summary["trajectory_guard_passed"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage18_9_trajectory_risk_boundary_reward_audit(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    coverage_comparison_root: Path | None = None,
    stage18_7_candidate_count_scaling_root: Path | None = None,
    canonical_profile: Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(
        config_path=config_path,
        repo_root=repo_root,
        overrides={
            "coverage_comparison_root": coverage_comparison_root,
            "stage18_7_candidate_count_scaling_root": stage18_7_candidate_count_scaling_root,
            "canonical_reward_profile": canonical_profile,
        },
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    profile = load_canonical_reward_profile(config["canonical_reward_profile"])
    if profile.profile_version != "v3":
        raise ConfigError("Stage 18.9 trajectory risk/reward audit requires canonical profile_version v3")
    thresholds = _trajectory_thresholds(profile)

    missing: list[str] = []
    blocking: list[str] = []
    diagnostic: list[str] = []
    coverage_root = Path(config["coverage_comparison_root"])
    coverage_summary = _read_json(coverage_root / COVERAGE_SUMMARY, missing, "missing_coverage_comparison_summary")
    aggregate = _read_json(coverage_root / COVERAGE_AGGREGATE, missing, "missing_coverage_comparison_aggregate")
    pairs = _read_jsonl(coverage_root / COVERAGE_PAIRS, missing, "missing_coverage_comparison_pairs")
    episodes = _read_jsonl(coverage_root / COVERAGE_EPISODES, missing, "missing_coverage_comparison_episodes")
    manifest = _read_json(coverage_root / COVERAGE_MANIFEST, missing, "missing_coverage_comparison_manifest")
    stage18_7 = _read_stage18_7(config)

    if _boundary_violations(config, coverage_summary, aggregate, manifest, stage18_7):
        blocking.append("boundary_violation")
    _validate_profile_lineage(
        expected={"profile_id": profile.profile_id, "profile_version": profile.profile_version, "profile_hash": profile.profile_hash},
        payloads=[coverage_summary, aggregate, manifest, *pairs],
        blocking=blocking,
    )
    if stage18_7 and _profile_identity(stage18_7).get("profile_hash") not in (None, profile.profile_hash):
        diagnostic.append("stage18_7_profile_lineage_diagnostic_only")

    scenario_rows = _scenario_rows(pairs, thresholds)
    boundary_summary = _boundary_summary(scenario_rows, episodes, thresholds)
    trajectory_summary = _trajectory_summary(scenario_rows, aggregate, coverage_summary, thresholds)
    reward_evaluation = _reward_evaluation(profile, scenario_rows, trajectory_summary)
    candidate_diagnostics = _candidate_diagnostics(stage18_7)

    if missing:
        blocking.extend(missing)
    if boundary_summary["hard_risk_violation_count"] > thresholds["max_hard_risk_violation_count"]:
        blocking.append("hard_risk_violation")
    if not trajectory_summary["coverage_advantage_established"]:
        blocking.append("coverage_advantage_not_established")
    if not trajectory_summary["path_cost_budget_passed"]:
        blocking.append("path_cost_budget_exceeded")
    if not trajectory_summary["coverage_efficiency_passed"]:
        blocking.append("coverage_efficiency_regression")
    if not trajectory_summary["soft_risk_exposure_passed"]:
        blocking.append("soft_risk_exposure_budget_exceeded")

    trajectory_guard_passed = not blocking
    route = _route(blocking)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "primary_route": route,
        "stage19_authorized": False,
    }
    readiness = {
        "schema_version": "xunce-stage18-9-stage19-readiness/v1",
        "readiness": "ready_for_stage19_preflight_human_review_only"
        if route == ROUTE_PREFLIGHT
        else "not_authorized",
        "authorized": False,
        "trajectory_guard_passed": trajectory_guard_passed,
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": "passed" if trajectory_guard_passed else "failed",
        "reason_codes": unique_sorted([*blocking, *diagnostic]),
        "blocking_reason_codes": unique_sorted(blocking),
        "diagnostic_reason_codes": unique_sorted(diagnostic),
        "coverage_comparison_root": str(coverage_root.resolve()),
        "stage18_7_candidate_count_scaling_root": str(Path(config["stage18_7_candidate_count_scaling_root"]).resolve())
        if config.get("stage18_7_candidate_count_scaling_root")
        else None,
        "canonical_reward_profile": str(Path(config["canonical_reward_profile"]).resolve()),
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "trajectory_thresholds": thresholds,
        "trajectory_guard_passed": trajectory_guard_passed,
        "trajectory_gate_passed": trajectory_guard_passed,
        "path_risk_boundary_summary": boundary_summary,
        "trajectory_guard_summary": trajectory_summary,
        "reward_profile_v3_evaluation": reward_evaluation,
        "candidate_diagnostics": candidate_diagnostics,
        "next_stage_routing": routing,
        "next_required_change": route,
        "stage19_readiness": readiness,
        "stage19_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "default_policy_replacement_approved": False,
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "summary": str(output_root / SUMMARY_FILE),
    }

    _write_jsonl(output_root / SCENARIO_AUDIT_FILE, scenario_rows)
    _write_json(output_root / BOUNDARY_FILE, boundary_summary)
    _write_json(output_root / REWARD_FILE, reward_evaluation)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    manifest_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": summary["generated_at"],
        "summary": str(output_root / SUMMARY_FILE),
        "artifacts": [
            SUMMARY_FILE,
            SCENARIO_AUDIT_FILE,
            BOUNDARY_FILE,
            REWARD_FILE,
            ROUTING_FILE,
            REPORT_FILE,
            MANIFEST_FILE,
        ],
        "stage19_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
    }
    _write_json(output_root / MANIFEST_FILE, manifest_payload)
    return summary


def _load_config(
    *,
    config_path: Path,
    repo_root: Path,
    overrides: dict[str, Path | None],
) -> dict[str, Any]:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file does not exist: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    for key, value in overrides.items():
        if value is not None:
            config[key] = str(value)
    for key in ("coverage_comparison_root", "canonical_reward_profile"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{key} must be a non-empty path string")
        config[key] = str(resolve_path(Path(value), repo_root).resolve())
    stage18_7_root = config.get("stage18_7_candidate_count_scaling_root")
    if stage18_7_root is not None:
        if not isinstance(stage18_7_root, str) or not stage18_7_root:
            raise ConfigError("stage18_7_candidate_count_scaling_root must be a non-empty path string")
        config["stage18_7_candidate_count_scaling_root"] = str(resolve_path(Path(stage18_7_root), repo_root).resolve())
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        raise ConfigError("canary_traffic_fraction must be 0.0")
    return config


def _trajectory_thresholds(profile: Any) -> dict[str, float]:
    guards = profile.guards
    return {
        "min_coverage_delta_cells": float(guards.get("min_coverage_delta_cells", 1.0)),
        "max_acceptable_path_cost_delta_m": float(guards.get("max_acceptable_path_cost_delta_m", 20.0)),
        "min_coverage_per_100m_delta": float(guards.get("min_coverage_per_100m_delta", 0.0)),
        "max_soft_risk_exposure_delta": float(
            guards.get("max_soft_risk_exposure_delta", guards.get("max_acceptable_risk_cost_weighted_delta", 25.0))
        ),
        "max_hard_risk_violation_count": float(guards.get("max_hard_risk_violation_count", 0.0)),
    }


def _scenario_rows(pairs: list[dict[str, Any]], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for index, pair in enumerate(pairs):
        observed = _observed_from_pair(pair)
        reasons = _scenario_failure_reasons(observed, thresholds)
        rows.append(
            {
                "schema_version": SCENARIO_ROW_SCHEMA_VERSION,
                "row_index": index,
                "scenario_id": pair.get("scenario_id"),
                "observed": observed,
                "trajectory_guard_passed": not reasons,
                "reason_codes": reasons,
            }
        )
    return rows


def _observed_from_pair(pair: dict[str, Any]) -> dict[str, float]:
    return {
        "coverage_delta_cells": _metric(pair, "coverage_delta_cells", "coverage_delta_cells_mean"),
        "path_cost_delta_m": _metric(pair, "path_cost_delta_m", "path_cost_delta_m_mean"),
        "coverage_per_100m_delta": _metric(pair, "coverage_per_100m_delta", "coverage_per_100m_delta_mean"),
        "soft_risk_exposure_delta": _metric(
            pair,
            "soft_risk_exposure_delta",
            "risk_cost_weighted_delta",
            "risk_delta",
        ),
        "hard_risk_violation_count": _metric(pair, "hard_risk_violation_count", "risk_boundary_violation_steps"),
    }


def _scenario_failure_reasons(observed: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if observed["hard_risk_violation_count"] > thresholds["max_hard_risk_violation_count"]:
        reasons.append("hard_risk_violation")
    if observed["coverage_delta_cells"] < thresholds["min_coverage_delta_cells"]:
        reasons.append("coverage_advantage_not_established")
    if observed["path_cost_delta_m"] > thresholds["max_acceptable_path_cost_delta_m"]:
        reasons.append("path_cost_budget_exceeded")
    if observed["coverage_per_100m_delta"] < thresholds["min_coverage_per_100m_delta"]:
        reasons.append("coverage_efficiency_regression")
    if observed["soft_risk_exposure_delta"] > thresholds["max_soft_risk_exposure_delta"]:
        reasons.append("soft_risk_exposure_budget_exceeded")
    return reasons


def _boundary_summary(rows: list[dict[str, Any]], episodes: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    hard_count = sum(_metric(row.get("observed", {}), "hard_risk_violation_count") for row in rows)
    if episodes:
        hard_count = max(hard_count, sum(_metric(row, "hard_risk_violation_count", "risk_boundary_violation_steps") for row in episodes))
    return {
        "schema_version": "xunce-stage18-9-path-risk-boundary-summary/v1",
        "path_risk_boundary_passed": hard_count <= thresholds["max_hard_risk_violation_count"],
        "hard_risk_violation_count": hard_count,
        "max_hard_risk_violation_count": thresholds["max_hard_risk_violation_count"],
        "risk_proxy_is_physical_risk": False,
    }


def _trajectory_summary(
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    coverage_summary: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    observed = {
        "coverage_delta_cells": _aggregate_or_rows(aggregate, coverage_summary, rows, "coverage_delta_cells", "coverage_delta_cells_mean"),
        "path_cost_delta_m": _aggregate_or_rows(aggregate, coverage_summary, rows, "path_cost_delta_m", "path_cost_delta_m_mean"),
        "coverage_per_100m_delta": _aggregate_or_rows(aggregate, coverage_summary, rows, "coverage_per_100m_delta", "coverage_per_100m_delta_mean"),
        "soft_risk_exposure_delta": _aggregate_or_rows(
            aggregate,
            coverage_summary,
            rows,
            "soft_risk_exposure_delta",
            "soft_risk_exposure_delta_mean",
            fallback_field="risk_cost_weighted_delta_mean",
        ),
    }
    return {
        "schema_version": "xunce-stage18-9-trajectory-guard-summary/v1",
        "observed": observed,
        "coverage_advantage_established": observed["coverage_delta_cells"] >= thresholds["min_coverage_delta_cells"],
        "path_cost_budget_passed": observed["path_cost_delta_m"] <= thresholds["max_acceptable_path_cost_delta_m"],
        "coverage_efficiency_passed": observed["coverage_per_100m_delta"] >= thresholds["min_coverage_per_100m_delta"],
        "soft_risk_exposure_passed": observed["soft_risk_exposure_delta"] <= thresholds["max_soft_risk_exposure_delta"],
    }


def _reward_evaluation(profile: Any, rows: list[dict[str, Any]], trajectory_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage18-9-reward-profile-v3-evaluation/v1",
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "reward_scope": "trajectory",
        "candidate_level_risk_delta_guard_is_diagnostic_only": True,
        "path_cost_includes_risk_proxy": bool((profile.risk_policy or {}).get("path_cost_includes_risk_proxy", False)),
        "component_set": [
            "coverage_component",
            "roi_coverage_component",
            "information_component",
            "path_cost_component",
            "soft_risk_component",
            "fallback_component",
            "failure_component",
        ],
        "trajectory_observed": trajectory_summary["observed"],
        "scenario_count": len(rows),
    }


def _candidate_diagnostics(stage18_7: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage18-9-candidate-diagnostics/v1",
        "diagnostic_only": True,
        "stage18_7_status": stage18_7.get("status"),
        "best_guard_clean_candidate_available_rate": stage18_7.get("best_guard_clean_candidate_available_rate"),
        "best_xunce_selected_guard_clean_rate": stage18_7.get("best_xunce_selected_guard_clean_rate"),
        "best_incumbent_selected_guard_clean_rate": stage18_7.get("best_incumbent_selected_guard_clean_rate"),
        "stage18_7_next_required_change": stage18_7.get("next_required_change")
        or (stage18_7.get("next_stage_routing") or {}).get("primary_route"),
    }


def _route(blocking: list[str]) -> str:
    blockers = set(blocking)
    if "boundary_violation" in blockers:
        return ROUTE_BOUNDARY
    if any(reason.startswith("missing_") or reason.startswith("invalid_") or reason in {"profile_hash_mismatch", "profile_id_mismatch", "profile_version_mismatch"} for reason in blockers):
        return ROUTE_INPUTS
    if "hard_risk_violation" in blockers:
        return ROUTE_RISK
    if "path_cost_budget_exceeded" in blockers or "coverage_efficiency_regression" in blockers or "coverage_advantage_not_established" in blockers:
        return ROUTE_REWARD
    if "soft_risk_exposure_budget_exceeded" in blockers:
        return ROUTE_SOFT_RISK
    return ROUTE_PREFLIGHT


def _validate_profile_lineage(*, expected: dict[str, str], payloads: list[dict[str, Any]], blocking: list[str]) -> None:
    missing_lineage = False
    mismatched_lineage = False
    for payload in payloads:
        if not isinstance(payload, dict) or not payload:
            continue
        identity = _profile_identity(payload)
        for field, expected_value in expected.items():
            value = identity.get(field)
            if value is None:
                missing_lineage = True
                blocking.append(f"{field}_missing")
            elif value != expected_value:
                mismatched_lineage = True
                blocking.append(f"{field}_mismatch")
    if missing_lineage:
        blocking.append("missing_profile_lineage")
    if mismatched_lineage:
        blocking.append("profile_lineage_mismatch")


def _profile_identity(payload: dict[str, Any]) -> dict[str, str | None]:
    return {
        "profile_id": payload.get("profile_id") or payload.get("canonical_guard_profile_id"),
        "profile_version": payload.get("profile_version") or payload.get("canonical_guard_profile_version"),
        "profile_hash": payload.get("profile_hash") or payload.get("canonical_guard_profile_hash"),
    }


def _read_stage18_7(config: dict[str, Any]) -> dict[str, Any]:
    root = config.get("stage18_7_candidate_count_scaling_root")
    if not root:
        return {}
    return _read_optional_json(Path(root) / STAGE18_7_SUMMARY)


def _read_json(path: Path, missing: list[str], reason: str) -> dict[str, Any]:
    if not path.is_file():
        missing.append(reason)
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        missing.append(f"invalid_{reason}")
        return {}
    return value if isinstance(value, dict) else {}


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path, missing: list[str], reason: str) -> list[dict[str, Any]]:
    if not path.is_file():
        missing.append(reason)
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            missing.append(f"invalid_{reason}")
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _boundary_violations(*payloads: dict[str, Any]) -> list[str]:
    violations = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for field in BOUNDARY_FIELDS:
            if payload.get(field) is True:
                violations.append(field)
        if float(payload.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
            violations.append("canary_traffic_fraction")
    return unique_sorted(violations)


def _aggregate_or_rows(
    aggregate: dict[str, Any],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    field: str,
    aggregate_field: str,
    *,
    fallback_field: str | None = None,
) -> float:
    for source, key in ((aggregate, aggregate_field), (summary, field), (aggregate, fallback_field)):
        if not key:
            continue
        value = _finite(source.get(key))
        if value is not None:
            return value
    values = [_finite((row.get("observed") or {}).get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else 0.0


def _metric(payload: dict[str, Any], *fields: str) -> float:
    for field in fields:
        value = _finite(payload.get(field))
        if value is not None:
            return value
    return 0.0


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _render_report(summary: dict[str, Any]) -> str:
    trajectory = summary["trajectory_guard_summary"]
    boundary = summary["path_risk_boundary_summary"]
    return "\n".join(
        [
            "# Xunce Stage 18.9 Trajectory Risk Boundary Reward Audit",
            "",
            f"- status: `{summary['status']}`",
            f"- trajectory_guard_passed: `{summary['trajectory_guard_passed']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- stage19_authorized: `{summary['stage19_authorized']}`",
            f"- hard_risk_violation_count: `{boundary['hard_risk_violation_count']}`",
            f"- coverage_advantage_established: `{trajectory['coverage_advantage_established']}`",
            f"- path_cost_budget_passed: `{trajectory['path_cost_budget_passed']}`",
            f"- coverage_efficiency_passed: `{trajectory['coverage_efficiency_passed']}`",
            f"- soft_risk_exposure_passed: `{trajectory['soft_risk_exposure_passed']}`",
            "",
            "Stage 18.9 treats candidate-level guard-clean metrics as diagnostics only. The readiness decision is based on the whole trajectory: hard risk boundary, final coverage, total path cost, coverage efficiency, and soft risk exposure.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
