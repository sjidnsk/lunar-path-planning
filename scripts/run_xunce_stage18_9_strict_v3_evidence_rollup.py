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


CONFIG_SCHEMA_VERSION = "xunce-stage18-9-strict-v3-evidence-rollup-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage18-9-strict-v3-evidence-summary/v1"
COUNT_ROW_SCHEMA_VERSION = "xunce-stage18-9-strict-v3-count-result-row/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage18-9-strict-v3-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage18-9-strict-v3-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage18_9_strict_v3_evidence_rollup_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage18_9_strict_v3_evidence_run/"
    "outputs/path_feedback_batch_xunce_stage18_9_strict_v3_evidence_rollup_v1"
)
DEFAULT_COUNTS = ((6, 48), (12, 96), (24, 192), (36, 288))

STAGE18_7_SUMMARY_FILE = "xunce-stage18-7-candidate-count-scaling-summary.json"
STAGE18_9_SUMMARY_FILE = "xunce-stage18-9-trajectory-risk-reward-summary.json"

SUMMARY_FILE = "xunce-stage18-9-strict-v3-evidence-summary.json"
COUNT_RESULTS_FILE = "xunce-stage18-9-strict-v3-count-results.jsonl"
ROUTING_FILE = "xunce-stage18-9-strict-v3-next-stage-routing.json"
REPORT_FILE = "xunce-stage18-9-strict-v3-report.md"
MANIFEST_FILE = "xunce-stage18-9-strict-v3-manifest.json"

ROUTE_BOUNDARY = "resolve_stage18_9_strict_v3_boundary_rejections"
ROUTE_INPUTS = "rerun_stage18_9_strict_v3_required_inputs"
ROUTE_RISK = "repair_path_risk_boundary_filtering"
ROUTE_REWARD = "refine_coverage_cost_reward_weights"
ROUTE_SOFT_RISK = "calibrate_soft_risk_exposure_weight"
ROUTE_PREFLIGHT = "prepare_stage19_evaluator_critic_preflight"

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "stage19_authorized",
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
    parser = argparse.ArgumentParser(description="Roll up strict v3 Stage 18.9 evidence across candidate counts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_stage18_9_strict_v3_evidence_rollup(
            config_path=resolve_path(Path(args.config), repo_root).resolve(),
            output_root=resolve_path(Path(args.output_root), repo_root).resolve(),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "best_candidate_count": summary.get("best_candidate_count"),
                "stage19_authorized": summary["stage19_authorized"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage18_9_strict_v3_evidence_rollup(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path=config_path, repo_root=repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    profile = load_canonical_reward_profile(config["canonical_reward_profile"])
    if profile.profile_version != "v3":
        raise ConfigError("strict v3 evidence rollup requires canonical profile_version v3")
    expected_identity = {
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
    }

    blocking: list[str] = []
    diagnostic: list[str] = []

    stage18_7_root = Path(config["stage18_7_candidate_count_scaling_root"])
    stage18_7_summary, stage18_7_reasons = _read_json(stage18_7_root / STAGE18_7_SUMMARY_FILE)
    blocking.extend(stage18_7_reasons)
    if stage18_7_summary:
        _validate_schema(stage18_7_summary, "xunce-stage18-7-candidate-count-scaling-summary/v1", blocking)
        _validate_profile_lineage(stage18_7_summary, expected_identity, blocking)
        if stage18_7_summary.get("status") not in {"passed", "partial"}:
            blocking.append("stage18_7_not_passed")
        blocking.extend(_boundary_violations(stage18_7_summary))

    rows: list[dict[str, Any]] = []
    for count_cfg in config["counts"]:
        row = _count_result_row(count_cfg, expected_identity)
        rows.append(row)
        blocking.extend(row["blocking_reason_codes"])
        diagnostic.extend(row["diagnostic_reason_codes"])

    blocking.extend(_boundary_violations(config))
    blocking = unique_sorted(blocking)
    diagnostic = unique_sorted(diagnostic)

    valid_rows = [row for row in rows if row["input_valid"]]
    hard_risk_rows = [row for row in valid_rows if row["hard_risk_violation_count"] > 0]
    passing_rows = [row for row in valid_rows if row["trajectory_guard_passed"] is True]
    if hard_risk_rows:
        blocking = unique_sorted([*blocking, "hard_risk_violation"])
    best_row = _best_row(passing_rows or valid_rows)
    route = _route(blocking, rows=valid_rows, passing_rows=passing_rows, hard_risk_rows=hard_risk_rows)
    trajectory_guard_passed_count = len(passing_rows)
    status = "passed" if route == ROUTE_PREFLIGHT and not blocking else "failed"

    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "primary_route": route,
        "stage19_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "reason_codes": unique_sorted([*blocking, *diagnostic]),
        "blocking_reason_codes": blocking,
        "diagnostic_reason_codes": diagnostic,
        "canonical_reward_profile": str(Path(config["canonical_reward_profile"]).resolve()),
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "artifact_workspace_root": str(Path(config["artifact_workspace_root"]).resolve())
        if config.get("artifact_workspace_root")
        else None,
        "stage18_7_candidate_count_scaling_root": str(stage18_7_root.resolve()),
        "count_result_count": len(rows),
        "valid_count_result_count": len(valid_rows),
        "trajectory_guard_passed_count": trajectory_guard_passed_count,
        "best_candidate_count": best_row.get("candidate_count") if best_row else None,
        "best_proposal_pool_limit": best_row.get("proposal_pool_limit") if best_row else None,
        "best_count_observed": best_row.get("observed") if best_row else None,
        "count_results": rows,
        "next_required_change": route,
        "next_stage_routing": routing,
        "stage19_authorized": False,
        "stage19_readiness": {
            "schema_version": "xunce-stage18-9-strict-v3-stage19-readiness/v1",
            "readiness": "ready_for_stage19_preflight_human_review_only"
            if route == ROUTE_PREFLIGHT
            else "not_authorized",
            "authorized": False,
            "trajectory_guard_passed_count": trajectory_guard_passed_count,
        },
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

    _write_jsonl(output_root / COUNT_RESULTS_FILE, rows)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    _write_json(
        output_root / MANIFEST_FILE,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at": summary["generated_at"],
            "summary": str(output_root / SUMMARY_FILE),
            "artifacts": [
                SUMMARY_FILE,
                COUNT_RESULTS_FILE,
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
        },
    )
    return summary


def _load_config(*, config_path: Path, repo_root: Path) -> dict[str, Any]:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")

    config = dict(payload)
    for key in ("artifact_workspace_root", "canonical_reward_profile", "stage18_7_candidate_count_scaling_root"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{key} must be a non-empty path string")
        config[key] = str(resolve_path(Path(value), repo_root).resolve())

    raw_counts = config.get("counts")
    if raw_counts is None:
        workspace = Path(config["artifact_workspace_root"])
        raw_counts = [
            {
                "candidate_count": count,
                "proposal_pool_limit": pool,
                "stage18_9_root": str(workspace / f"count_{count:03d}" / "stage18_9"),
            }
            for count, pool in DEFAULT_COUNTS
        ]
    if not isinstance(raw_counts, list) or not raw_counts:
        raise ConfigError("counts must be a non-empty array")

    counts: list[dict[str, Any]] = []
    for index, item in enumerate(raw_counts):
        if not isinstance(item, dict):
            raise ConfigError(f"counts[{index}] must be an object")
        count = _positive_int(item.get("candidate_count"), f"counts[{index}].candidate_count")
        pool = _positive_int(item.get("proposal_pool_limit"), f"counts[{index}].proposal_pool_limit")
        stage18_9_root = item.get("stage18_9_root")
        if not isinstance(stage18_9_root, str) or not stage18_9_root:
            raise ConfigError(f"counts[{index}].stage18_9_root must be a non-empty path string")
        counts.append(
            {
                "candidate_count": count,
                "proposal_pool_limit": pool,
                "stage18_9_root": str(resolve_path(Path(stage18_9_root), repo_root).resolve()),
            }
        )
    config["counts"] = counts

    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        raise ConfigError("canary_traffic_fraction must be 0.0")
    return config


def _count_result_row(count_cfg: dict[str, Any], expected_identity: dict[str, str]) -> dict[str, Any]:
    root = Path(count_cfg["stage18_9_root"])
    summary, read_reasons = _read_json(root / STAGE18_9_SUMMARY_FILE)
    blocking = list(read_reasons)
    diagnostic: list[str] = []
    if summary:
        _validate_schema(summary, "xunce-stage18-9-trajectory-risk-reward-summary/v1", blocking)
        _validate_profile_lineage(summary, expected_identity, blocking)
        blocking.extend(_boundary_violations(summary))
        blocking.extend(_authorization_reasons(summary))
    observed = _observed(summary)
    blocking.extend(_missing_observed_metric_reasons(summary, observed))
    hard_risk_violation_count = _path_risk_violation_count(summary, observed)
    if summary and not isinstance(summary.get("trajectory_guard_passed"), bool):
        blocking.append("invalid_trajectory_guard_passed")
    trajectory_guard_passed = summary.get("trajectory_guard_passed") is True if summary else False
    route = summary.get("next_required_change") or (summary.get("next_stage_routing") or {}).get("primary_route")
    if summary and route == ROUTE_PREFLIGHT and not _preflight_semantics_are_clean(summary, observed):
        blocking.append("invalid_stage18_9_preflight_semantics")
    if summary and hard_risk_violation_count > 0:
        diagnostic.append("hard_risk_violation")

    input_valid = not blocking and bool(summary)
    return {
        "schema_version": COUNT_ROW_SCHEMA_VERSION,
        "candidate_count": count_cfg["candidate_count"],
        "proposal_pool_limit": count_cfg["proposal_pool_limit"],
        "stage18_9_root": str(root.resolve()),
        "input_valid": input_valid,
        "profile_id": summary.get("profile_id"),
        "profile_version": summary.get("profile_version"),
        "profile_hash": summary.get("profile_hash"),
        "trajectory_guard_passed": trajectory_guard_passed if input_valid else False,
        "hard_risk_violation_count": hard_risk_violation_count,
        "coverage_delta_cells": observed["coverage_delta_cells"],
        "path_cost_delta_m": observed["path_cost_delta_m"],
        "coverage_per_100m_delta": observed["coverage_per_100m_delta"],
        "soft_risk_exposure_delta": observed["soft_risk_exposure_delta"],
        "observed": observed,
        "stage18_9_next_required_change": route,
        "stage19_authorized": bool(summary.get("stage19_authorized", False)),
        "blocking_reason_codes": unique_sorted(blocking),
        "diagnostic_reason_codes": unique_sorted(diagnostic),
    }


def _route(
    blocking: list[str],
    *,
    rows: list[dict[str, Any]],
    passing_rows: list[dict[str, Any]],
    hard_risk_rows: list[dict[str, Any]],
) -> str:
    blockers = set(blocking)
    if (
        "boundary_violation" in blockers
        or "canary_traffic_fraction" in blockers
        or "stage19_authorized_not_false" in blockers
        or any(reason in blockers for reason in BOUNDARY_FIELDS)
    ):
        return ROUTE_BOUNDARY
    input_blockers = {
        "missing_json",
        "invalid_json",
        "invalid_schema_version",
        "invalid_trajectory_guard_passed",
        "invalid_stage18_9_preflight_semantics",
        "missing_observed_metrics",
        "profile_id_mismatch",
        "profile_version_mismatch",
        "profile_hash_mismatch",
        "stage18_7_not_passed",
        "stage19_authorized_not_false",
    }
    if blockers.intersection(input_blockers) or not rows:
        return ROUTE_INPUTS
    if hard_risk_rows:
        return ROUTE_RISK
    if passing_rows:
        return ROUTE_PREFLIGHT
    if not any(row["coverage_delta_cells"] >= 1.0 for row in rows):
        return ROUTE_REWARD
    if any(row["path_cost_delta_m"] > 20.0 or row["coverage_per_100m_delta"] < 0.0 for row in rows):
        return ROUTE_REWARD
    if any(row["soft_risk_exposure_delta"] > 25.0 for row in rows):
        return ROUTE_SOFT_RISK
    return ROUTE_REWARD


def _observed(summary: dict[str, Any]) -> dict[str, float]:
    trajectory = summary.get("trajectory_guard_summary") if isinstance(summary, dict) else {}
    observed = trajectory.get("observed") if isinstance(trajectory, dict) else {}
    if not isinstance(observed, dict):
        observed = {}
    return {
        "coverage_delta_cells": _metric(observed, summary, "coverage_delta_cells"),
        "path_cost_delta_m": _metric(observed, summary, "path_cost_delta_m"),
        "coverage_per_100m_delta": _metric(observed, summary, "coverage_per_100m_delta"),
        "soft_risk_exposure_delta": _metric(
            observed,
            summary,
            "soft_risk_exposure_delta",
            fallback_keys=("risk_cost_weighted_delta", "risk_cost_weighted_delta_mean"),
        ),
    }


def _missing_observed_metric_reasons(summary: dict[str, Any], observed: dict[str, float]) -> list[str]:
    if not summary:
        return []
    missing = []
    trajectory = summary.get("trajectory_guard_summary")
    observed_payload = trajectory.get("observed") if isinstance(trajectory, dict) else {}
    if not isinstance(observed_payload, dict):
        missing.append("trajectory_guard_summary.observed")
        observed_payload = {}
    for field in ("coverage_delta_cells", "path_cost_delta_m", "coverage_per_100m_delta", "soft_risk_exposure_delta"):
        if _finite(observed_payload.get(field)) is None and _finite(summary.get(field)) is None:
            missing.append(field)
    boundary = summary.get("path_risk_boundary_summary")
    if (
        not isinstance(boundary, dict)
        or _finite(boundary.get("hard_risk_violation_count")) is None
    ) and _finite(summary.get("hard_risk_violation_count")) is None:
        missing.append("hard_risk_violation_count")
    if missing:
        return ["missing_observed_metrics", *[f"missing_stage18_9_observed_metric:{field}" for field in missing]]
    return []


def _path_risk_violation_count(summary: dict[str, Any], observed: dict[str, float]) -> float:
    boundary = summary.get("path_risk_boundary_summary") if isinstance(summary, dict) else {}
    if isinstance(boundary, dict):
        value = _finite(boundary.get("hard_risk_violation_count"))
        if value is not None:
            return value
    return _finite(summary.get("hard_risk_violation_count")) or _finite(observed.get("hard_risk_violation_count")) or 0.0


def _authorization_reasons(summary: dict[str, Any]) -> list[str]:
    reasons = []
    routing = summary.get("next_stage_routing")
    readiness = summary.get("stage19_readiness")
    if summary.get("stage19_authorized") is not False:
        reasons.append("stage19_authorized_not_false")
    if isinstance(routing, dict) and routing.get("stage19_authorized") is not False:
        reasons.append("stage19_authorized_not_false")
    if isinstance(readiness, dict) and readiness.get("authorized") is not False:
        reasons.append("stage19_authorized_not_false")
    return unique_sorted(reasons)


def _preflight_semantics_are_clean(summary: dict[str, Any], observed: dict[str, float]) -> bool:
    routing = summary.get("next_stage_routing")
    readiness = summary.get("stage19_readiness")
    trajectory = summary.get("trajectory_guard_summary")
    boundary = summary.get("path_risk_boundary_summary")
    if not (
        summary.get("status") == "passed"
        and summary.get("trajectory_guard_passed") is True
        and summary.get("stage19_authorized") is False
        and isinstance(routing, dict)
        and routing.get("stage19_authorized") is False
        and isinstance(readiness, dict)
        and readiness.get("authorized") is False
        and isinstance(boundary, dict)
        and boundary.get("path_risk_boundary_passed") is True
        and isinstance(trajectory, dict)
        and trajectory.get("coverage_advantage_established") is True
        and trajectory.get("path_cost_budget_passed") is True
        and trajectory.get("coverage_efficiency_passed") is True
        and trajectory.get("soft_risk_exposure_passed") is True
    ):
        return False
    return (
        observed["coverage_delta_cells"] >= 1.0
        and observed["path_cost_delta_m"] <= 20.0
        and observed["coverage_per_100m_delta"] >= 0.0
        and observed["soft_risk_exposure_delta"] <= 25.0
        and _path_risk_violation_count(summary, observed) == 0.0
    )


def _metric(primary: dict[str, Any], secondary: dict[str, Any], key: str, *, fallback_keys: tuple[str, ...] = ()) -> float:
    for source in (primary, secondary):
        for field in (key, *fallback_keys):
            value = _finite(source.get(field)) if isinstance(source, dict) else None
            if value is not None:
                return value
    return 0.0


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("coverage_per_100m_delta", 0.0) or 0.0),
            float(row.get("path_cost_delta_m", 0.0) or 0.0),
            -float(row.get("coverage_delta_cells", 0.0) or 0.0),
            int(row.get("candidate_count", 0) or 0),
        ),
    )[0]


def _read_json(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, ["missing_json"]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, ["invalid_json"]
    if not isinstance(value, dict):
        return {}, ["invalid_json"]
    return value, []


def _validate_schema(payload: dict[str, Any], expected_schema: str, blocking: list[str]) -> None:
    if payload.get("schema_version") != expected_schema:
        blocking.append("invalid_schema_version")


def _validate_profile_lineage(payload: dict[str, Any], expected: dict[str, str], blocking: list[str]) -> None:
    mismatched = False
    for field, expected_value in expected.items():
        value = payload.get(field) or payload.get(f"canonical_guard_{field}")
        if value != expected_value:
            mismatched = True
            blocking.append(f"{field}_mismatch")
    if mismatched:
        blocking.append("profile_lineage_mismatch")


def _boundary_violations(*payloads: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for field in BOUNDARY_FIELDS:
            if payload.get(field) is True:
                violations.append(field)
        if float(payload.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
            violations.append("canary_traffic_fraction")
    return unique_sorted(violations)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise ConfigError(f"{label} must be a positive integer")
    return parsed


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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Stage 18.9 Strict V3 Evidence Rollup",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- best_candidate_count: `{summary['best_candidate_count']}`",
            f"- trajectory_guard_passed_count: `{summary['trajectory_guard_passed_count']}`",
            f"- stage19_authorized: `{summary['stage19_authorized']}`",
            f"- profile_version: `{summary['profile_version']}`",
            f"- profile_hash: `{summary['profile_hash']}`",
            "",
            "This rollup only accepts strict v3 Stage 18.9 trajectory evidence. Stage 18.6/18.7 candidate clean metrics remain diagnostic; they do not authorize PPO, checkpoint publishing, default policy replacement, executor connection, or online canary traffic.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
