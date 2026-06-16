from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from run_global_99_multi_map_generalization import run_global_99_multi_map_generalization
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import (
        ConfigError,
        resolve_path,
        unique_sorted,
        utc_now,
        write_json,
        write_jsonl,
    )
    from scripts.run_global_99_multi_map_generalization import run_global_99_multi_map_generalization


CONFIG_SCHEMA_VERSION = "global-99-shadow-canary-replay-config/v1"
SUMMARY_SCHEMA_VERSION = "global-99-shadow-canary-replay-summary/v1"
MANIFEST_SCHEMA_VERSION = "global-99-shadow-canary-replay-manifest/v1"
SCENARIO_RESULT_SCHEMA_VERSION = "global-99-shadow-canary-replay-scenario-result/v1"
FAMILY_SUMMARY_SCHEMA_VERSION = "global-99-shadow-canary-replay-family-summary/v1"
POLICY_AUDIT_SCHEMA_VERSION = "global-99-shadow-canary-replay-policy-vs-baseline-audit/v1"
SOURCE_MATCH_AUDIT_SCHEMA_VERSION = "global-99-shadow-canary-replay-source-match-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "global-99-shadow-canary-replay-boundary-audit/v1"
KILL_SWITCH_AUDIT_SCHEMA_VERSION = "global-99-shadow-canary-replay-kill-switch-audit/v1"
ROLLBACK_AUDIT_SCHEMA_VERSION = "global-99-shadow-canary-replay-rollback-audit/v1"
TELEMETRY_AUDIT_SCHEMA_VERSION = "global-99-shadow-canary-replay-telemetry-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "global-99-shadow-canary-replay-rejection-report/v1"

DEFAULT_CONFIG = "configs/global_99_shadow_canary_replay_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_global_99_shadow_canary_replay_v1"

PREFLIGHT_SUMMARY_FILE = "global-99-shadow-canary-preflight-summary.json"
MULTI_MAP_SUMMARY_FILE = "global-99-multi-map-generalization-summary.json"
MULTI_MAP_SCENARIO_RESULTS_FILE = "global-99-multi-map-scenario-results.jsonl"
MULTI_MAP_FAMILY_SUMMARY_FILE = "global-99-multi-map-family-summary.json"
MULTI_MAP_POLICY_AUDIT_FILE = "global-99-multi-map-policy-vs-baseline-audit.json"

SUMMARY_FILE = "global-99-shadow-canary-replay-summary.json"
MANIFEST_FILE = "global-99-shadow-canary-replay-manifest.json"
SCENARIO_RESULTS_FILE = "global-99-shadow-canary-replay-scenario-results.jsonl"
FAMILY_SUMMARY_FILE = "global-99-shadow-canary-replay-family-summary.json"
SOURCE_MATCH_AUDIT_FILE = "global-99-shadow-canary-replay-source-match-audit.json"
POLICY_AUDIT_FILE = "global-99-shadow-canary-replay-policy-vs-baseline-audit.json"
BOUNDARY_AUDIT_FILE = "global-99-shadow-canary-replay-boundary-audit.json"
KILL_SWITCH_AUDIT_FILE = "global-99-shadow-canary-replay-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "global-99-shadow-canary-replay-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "global-99-shadow-canary-replay-telemetry-audit.json"
REJECTION_REPORT_FILE = "global-99-shadow-canary-replay-rejection-report.json"
REPORT_FILE = "global-99-shadow-canary-replay-report.md"

PREFLIGHT_PASS_VERDICT = "eligible_for_global_99_shadow_canary_replay"
PASS_NEXT_REQUIRED_CHANGE = "global_99_real_map_preflight"
FIX_PREFLIGHT_NEXT_REQUIRED_CHANGE = "fix_global_99_shadow_canary_preflight"
FIX_MULTI_MAP_NEXT_REQUIRED_CHANGE = "fix_global_99_multi_map_generalization"
FIX_REPLAY_NEXT_REQUIRED_CHANGE = "fix_global_99_shadow_canary_replay"
FIX_DETERMINISM_NEXT_REQUIRED_CHANGE = "fix_global_99_shadow_canary_replay_determinism"
FIX_POLICY_NEXT_REQUIRED_CHANGE = "fix_policy_guided_global_coverage"
FIX_FALLBACK_NEXT_REQUIRED_CHANGE = "fix_global_99_shadow_canary_guard_fallback"
BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE = "resolve_global_99_shadow_canary_replay_boundary_rejections"

BOUNDARY_FIELDS = {
    "uses_policy_guidance": True,
    "uses_checkpoint_inference": True,
    "policy_read_only": True,
    "publishes_checkpoint": False,
    "replaces_default_policy": False,
    "connects_real_executor": False,
    "runs_new_ppo_update": False,
    "modifies_network": False,
    "modifies_action_space": False,
    "modifies_default_astar": False,
    "uses_path_planner": False,
    "uses_npz_or_sidecar": False,
    "real_world_release_approved": False,
    "real_world_performance_claimed": False,
    "default_policy_replacement_approved": False,
    "real_executor_connection_approved": False,
    "starts_online_canary": False,
}

FORBIDDEN_BOUNDARY_FIELDS = tuple(key for key, value in BOUNDARY_FIELDS.items() if value is False)

REPLAY_BOUNDARY_FIELDS = FORBIDDEN_BOUNDARY_FIELDS + (
    "checkpoint_publication_approved",
    "final_release_approved",
    "relaxes_guard",
    "guard_relaxed",
    "modifies_network_or_action_space",
    "ackermann_feasible_trajectory_claimed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Global 99 Shadow Canary Replay v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_global_99_shadow_canary_replay(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "shadow_replay_passed": summary["shadow_replay_passed"],
                "offline_canary_replay_passed": summary["offline_canary_replay_passed"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_global_99_shadow_canary_replay(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config_path = Path(config_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(config_path, repo_root)
    source_paths = _source_paths(config, repo_root)
    paths = _artifact_paths(output_root)
    evidence = _load_evidence(source_paths)
    replay_payload = _run_replay_matrix(config, config_path, output_root, repo_root)
    replay_results = [_replay_scenario_row(row) for row in replay_payload["scenario_results"]]
    family_summary = _replay_family_summary(replay_payload["family_summary"])
    policy_audit = _replay_policy_audit(replay_payload["policy_audit"])

    source_match = _source_match_audit(config, evidence, replay_results)
    boundary = _boundary_audit(config, evidence, policy_audit)
    telemetry = _telemetry_audit(policy_audit)
    replay_audit = _shadow_replay_audit(config, evidence, replay_results, source_match, boundary)
    canary_audit = _offline_canary_replay_audit(config, evidence, policy_audit, source_match, boundary)
    kill_switch = _kill_switch_audit(source_match, boundary)
    rollback = _rollback_audit(source_match, boundary)
    decision = _decision(
        config,
        evidence,
        replay_results,
        source_match,
        policy_audit,
        replay_audit,
        canary_audit,
        boundary,
        kill_switch,
        rollback,
        telemetry,
    )
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        source_paths=source_paths,
        evidence=evidence,
        replay_results=replay_results,
        source_match=source_match,
        policy_audit=policy_audit,
        replay_audit=replay_audit,
        canary_audit=canary_audit,
        boundary=boundary,
        kill_switch=kill_switch,
        rollback=rollback,
        telemetry=telemetry,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)
    rejection_report = _rejection_report(generated_at, decision, boundary, source_match)

    write_jsonl(paths["scenario_results"], replay_results)
    write_json(paths["family_summary"], family_summary)
    write_json(paths["source_match_audit"], source_match)
    write_json(paths["policy_vs_baseline_audit"], policy_audit)
    write_json(paths["boundary"], boundary)
    write_json(paths["kill_switch"], kill_switch)
    write_json(paths["rollback"], rollback)
    write_json(paths["telemetry"], telemetry)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, rejection_report), encoding="utf-8")
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for key in (
        "source_shadow_canary_preflight_root",
        "source_multi_map_root",
        "source_policy_guided_config",
        "source_multi_map_config",
    ):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    normalized = dict(payload)
    normalized["source_shadow_canary_preflight_root"] = str(
        resolve_path(Path(payload["source_shadow_canary_preflight_root"]), repo_root)
    )
    normalized["source_multi_map_root"] = str(resolve_path(Path(payload["source_multi_map_root"]), repo_root))
    normalized["source_policy_guided_config"] = str(resolve_path(Path(payload["source_policy_guided_config"]), repo_root))
    normalized["source_multi_map_config"] = str(resolve_path(Path(payload["source_multi_map_config"]), repo_root))
    normalized["target_coverage_rate"] = _bounded_float(payload.get("target_coverage_rate"), "target_coverage_rate")
    normalized["max_policy_guard_fallback_rate"] = _bounded_float(
        payload.get("max_policy_guard_fallback_rate"),
        "max_policy_guard_fallback_rate",
    )
    normalized["replay_coverage_tolerance"] = _non_negative_float(
        payload.get("replay_coverage_tolerance"),
        "replay_coverage_tolerance",
    )
    normalized["replay_path_cost_tolerance_m"] = _non_negative_float(
        payload.get("replay_path_cost_tolerance_m"),
        "replay_path_cost_tolerance_m",
    )
    normalized["shadow_replay_mode"] = _bool_value(payload.get("shadow_replay_mode"), "shadow_replay_mode")
    normalized["offline_canary_replay_mode"] = _bool_value(
        payload.get("offline_canary_replay_mode"),
        "offline_canary_replay_mode",
    )
    normalized["canary_traffic_fraction"] = _fraction_float(
        payload.get("canary_traffic_fraction"),
        "canary_traffic_fraction",
    )
    normalized["require_policy_read_only"] = _bool_value(payload.get("require_policy_read_only"), "require_policy_read_only")
    return normalized


def _source_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    preflight_root = resolve_path(Path(config["source_shadow_canary_preflight_root"]), repo_root)
    multi_map_root = resolve_path(Path(config["source_multi_map_root"]), repo_root)
    return {
        "shadow_canary_preflight_root": preflight_root,
        "multi_map_root": multi_map_root,
        "shadow_canary_preflight_summary": preflight_root / PREFLIGHT_SUMMARY_FILE,
        "multi_map_summary": multi_map_root / MULTI_MAP_SUMMARY_FILE,
        "multi_map_scenario_results": multi_map_root / MULTI_MAP_SCENARIO_RESULTS_FILE,
        "multi_map_family_summary": multi_map_root / MULTI_MAP_FAMILY_SUMMARY_FILE,
        "multi_map_policy_vs_baseline_audit": multi_map_root / MULTI_MAP_POLICY_AUDIT_FILE,
    }


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "scenario_results": output_root / SCENARIO_RESULTS_FILE,
        "family_summary": output_root / FAMILY_SUMMARY_FILE,
        "source_match_audit": output_root / SOURCE_MATCH_AUDIT_FILE,
        "policy_vs_baseline_audit": output_root / POLICY_AUDIT_FILE,
        "boundary": output_root / BOUNDARY_AUDIT_FILE,
        "kill_switch": output_root / KILL_SWITCH_AUDIT_FILE,
        "rollback": output_root / ROLLBACK_AUDIT_FILE,
        "telemetry": output_root / TELEMETRY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _load_evidence(source_paths: dict[str, Path]) -> dict[str, Any]:
    read_reasons: list[str] = []
    return {
        "shadow_canary_preflight": _read_json(
            source_paths["shadow_canary_preflight_summary"],
            read_reasons,
            "shadow_canary_preflight_summary",
        ),
        "multi_map_summary": _read_json(source_paths["multi_map_summary"], read_reasons, "multi_map_summary"),
        "multi_map_family_summary": _read_json(
            source_paths["multi_map_family_summary"],
            read_reasons,
            "multi_map_family_summary",
        ),
        "multi_map_policy_audit": _read_json(
            source_paths["multi_map_policy_vs_baseline_audit"],
            read_reasons,
            "multi_map_policy_audit",
        ),
        "multi_map_scenario_results": _read_jsonl(
            source_paths["multi_map_scenario_results"],
            read_reasons,
            "multi_map_scenario_results",
        ),
        "read_reason_codes": unique_sorted(read_reasons),
    }


def _read_json(path: Path, reasons: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(f"missing_{label}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        reasons.append(f"invalid_{label}")
        return {}
    if not isinstance(payload, dict):
        reasons.append(f"invalid_{label}")
        return {}
    return payload


def _read_jsonl(path: Path, reasons: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        reasons.append(f"missing_{label}")
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                reasons.append(f"invalid_{label}")
                return []
            rows.append(row)
    except json.JSONDecodeError:
        reasons.append(f"invalid_{label}")
        return []
    return rows


def _run_replay_matrix(
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    replay_work_root = output_root / "_multi_map_replay_work"
    summary = run_global_99_multi_map_generalization(
        config_path=Path(config["source_multi_map_config"]),
        output_root=replay_work_root,
        repo_root=repo_root,
    )
    scenario_results = _read_jsonl(replay_work_root / MULTI_MAP_SCENARIO_RESULTS_FILE, [], "replay_scenario_results")
    family_summary = _read_json(replay_work_root / MULTI_MAP_FAMILY_SUMMARY_FILE, [], "replay_family_summary")
    policy_audit = _read_json(replay_work_root / MULTI_MAP_POLICY_AUDIT_FILE, [], "replay_policy_audit")
    policy_audit["replay_multi_map_summary_status"] = summary.get("status")
    return {
        "scenario_results": scenario_results,
        "family_summary": family_summary,
        "policy_audit": policy_audit,
    }


def _replay_scenario_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["schema_version"] = SCENARIO_RESULT_SCHEMA_VERSION
    return normalized


def _replay_family_summary(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["schema_version"] = FAMILY_SUMMARY_SCHEMA_VERSION
    return normalized


def _replay_policy_audit(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["schema_version"] = POLICY_AUDIT_SCHEMA_VERSION
    guided = int(normalized.get("policy_guided_decision_count", 0))
    fallbacks = int(normalized.get("policy_guard_fallback_count", 0))
    normalized["policy_guard_fallback_rate"] = fallbacks / guided if guided else 1.0
    normalized["policy_loaded"] = bool(normalized.get("policy_loaded", False))
    normalized["policy_guidance_applied"] = bool(normalized.get("policy_guidance_applied", False))
    return normalized


def _source_match_audit(
    config: dict[str, Any],
    evidence: dict[str, Any],
    replay_results: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons = []
    if evidence["read_reason_codes"]:
        reasons.append("source_match_evidence_incomplete")
    source_by_id = {row.get("scenario_id"): row for row in evidence["multi_map_scenario_results"] if row.get("scenario_id")}
    replay_by_id = {row.get("scenario_id"): row for row in replay_results if row.get("scenario_id")}
    missing_source = sorted(set(replay_by_id) - set(source_by_id))
    missing_replay = sorted(set(source_by_id) - set(replay_by_id))
    comparisons = []
    max_coverage_delta = 0.0
    max_path_delta = 0.0
    for scenario_id in sorted(set(source_by_id) & set(replay_by_id)):
        source = source_by_id[scenario_id]
        replay = replay_by_id[scenario_id]
        coverage_delta = abs(float(source.get("achieved_coverage_rate", 0.0)) - float(replay.get("achieved_coverage_rate", 0.0)))
        path_delta = abs(float(source.get("planned_path_cost_m", 0.0)) - float(replay.get("planned_path_cost_m", 0.0)))
        max_coverage_delta = max(max_coverage_delta, coverage_delta)
        max_path_delta = max(max_path_delta, path_delta)
        coverage_match = coverage_delta <= float(config["replay_coverage_tolerance"])
        path_match = path_delta <= float(config["replay_path_cost_tolerance_m"])
        if not coverage_match or not path_match:
            reasons.append("shadow_replay_source_mismatch")
        comparisons.append(
            {
                "scenario_id": scenario_id,
                "source_achieved_coverage_rate": float(source.get("achieved_coverage_rate", 0.0)),
                "replay_achieved_coverage_rate": float(replay.get("achieved_coverage_rate", 0.0)),
                "coverage_delta": coverage_delta,
                "source_planned_path_cost_m": float(source.get("planned_path_cost_m", 0.0)),
                "replay_planned_path_cost_m": float(replay.get("planned_path_cost_m", 0.0)),
                "path_cost_delta_m": path_delta,
                "coverage_match": coverage_match,
                "path_cost_match": path_match,
            }
        )
    if missing_source:
        reasons.append("shadow_replay_source_missing_scenarios")
    if missing_replay:
        reasons.append("shadow_replay_missing_replay_scenarios")
    return {
        "schema_version": SOURCE_MATCH_AUDIT_SCHEMA_VERSION,
        "source_match_audit_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "source_scenario_count": len(source_by_id),
        "replay_scenario_count": len(replay_by_id),
        "missing_source_scenario_ids": missing_source,
        "missing_replay_scenario_ids": missing_replay,
        "max_replay_coverage_delta": max_coverage_delta,
        "max_replay_path_cost_delta_m": max_path_delta,
        "replay_coverage_tolerance": config["replay_coverage_tolerance"],
        "replay_path_cost_tolerance_m": config["replay_path_cost_tolerance_m"],
        "scenario_comparisons": comparisons,
    }


def _boundary_audit(
    config: dict[str, Any],
    evidence: dict[str, Any],
    policy_audit: dict[str, Any],
) -> dict[str, Any]:
    violations = []
    for source_name in ("shadow_canary_preflight", "multi_map_summary", "multi_map_policy_audit"):
        payload = evidence[source_name]
        for field in REPLAY_BOUNDARY_FIELDS:
            if payload.get(field) is True:
                violations.append({"source": source_name, "field": field, "value": True})
        if float(payload.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
            violations.append(
                {
                    "source": source_name,
                    "field": "canary_traffic_fraction",
                    "value": payload.get("canary_traffic_fraction"),
                }
            )
    for field in REPLAY_BOUNDARY_FIELDS:
        if policy_audit.get(field) is True:
            violations.append({"source": "replay_policy_audit", "field": field, "value": True})
    if float(config["canary_traffic_fraction"]) != 0.0:
        violations.append({"source": "config", "field": "canary_traffic_fraction", "value": config["canary_traffic_fraction"]})
    return {
        "schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION,
        "boundary_audit_passed": not violations,
        "reason_codes": [] if not violations else ["shadow_canary_replay_boundary_violation"],
        "violations": violations,
        **BOUNDARY_FIELDS,
        "canary_traffic_fraction": 0.0,
    }


def _telemetry_audit(policy_audit: dict[str, Any]) -> dict[str, Any]:
    passed = (
        policy_audit["policy_loaded"]
        and policy_audit["policy_guidance_applied"]
        and int(policy_audit.get("policy_scored_candidate_count", 0)) > 0
        and int(policy_audit.get("policy_guided_decision_count", 0)) > 0
    )
    return {
        "schema_version": TELEMETRY_AUDIT_SCHEMA_VERSION,
        "telemetry_audit_passed": passed,
        "reason_codes": [] if passed else ["telemetry_evidence_incomplete"],
        "tracked_counters": {
            "policy_scored_candidate_count": int(policy_audit.get("policy_scored_candidate_count", 0)),
            "policy_guided_decision_count": int(policy_audit.get("policy_guided_decision_count", 0)),
            "policy_guard_fallback_count": int(policy_audit.get("policy_guard_fallback_count", 0)),
        },
    }


def _shadow_replay_audit(
    config: dict[str, Any],
    evidence: dict[str, Any],
    replay_results: list[dict[str, Any]],
    source_match: dict[str, Any],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    reasons = []
    if not config["shadow_replay_mode"]:
        reasons.append("shadow_replay_mode_disabled")
    if evidence["shadow_canary_preflight"].get("status") != "passed":
        reasons.append("shadow_canary_preflight_not_passed")
    if _failed_required_count(replay_results) > 0:
        reasons.append("shadow_replay_required_scenarios_not_passed")
    if not source_match["source_match_audit_passed"]:
        reasons.append("shadow_replay_source_match_blocked")
    if not boundary["boundary_audit_passed"]:
        reasons.append("shadow_replay_boundary_blocked")
    return {
        "schema_version": "global-99-shadow-canary-shadow-replay-audit/v1",
        "shadow_replay_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "shadow_replay_mode": config["shadow_replay_mode"],
        "shadow_replay_real_executor_connected": False,
    }


def _offline_canary_replay_audit(
    config: dict[str, Any],
    evidence: dict[str, Any],
    policy_audit: dict[str, Any],
    source_match: dict[str, Any],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    reasons = []
    if not config["offline_canary_replay_mode"]:
        reasons.append("offline_canary_replay_mode_disabled")
    if float(config["canary_traffic_fraction"]) != 0.0:
        reasons.append("canary_traffic_fraction_must_be_zero")
    if config["require_policy_read_only"] and evidence["multi_map_summary"].get("policy_read_only") is not True:
        reasons.append("policy_read_only_not_verified")
    if not policy_audit["policy_guidance_applied"]:
        reasons.append("policy_guidance_not_applied")
    if not source_match["source_match_audit_passed"]:
        reasons.append("offline_canary_source_match_blocked")
    if not boundary["boundary_audit_passed"]:
        reasons.append("offline_canary_boundary_blocked")
    return {
        "schema_version": "global-99-shadow-canary-offline-canary-replay-audit/v1",
        "offline_canary_replay_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "offline_canary_replay_mode": config["offline_canary_replay_mode"],
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _kill_switch_audit(source_match: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    passed = source_match["source_match_audit_passed"] and boundary["boundary_audit_passed"]
    return {
        "schema_version": KILL_SWITCH_AUDIT_SCHEMA_VERSION,
        "kill_switch_audit_passed": passed,
        "reason_codes": [] if passed else ["kill_switch_contract_blocked"],
        "kill_switch_required_before_real_map_or_online_canary": True,
        "kill_switch_real_control_plane_connected": False,
    }


def _rollback_audit(source_match: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    passed = source_match["source_match_audit_passed"] and boundary["boundary_audit_passed"]
    return {
        "schema_version": ROLLBACK_AUDIT_SCHEMA_VERSION,
        "rollback_audit_passed": passed,
        "reason_codes": [] if passed else ["rollback_contract_blocked"],
        "rollback_required_before_real_map_or_online_canary": True,
        "rollback_real_control_plane_connected": False,
    }


def _decision(
    config: dict[str, Any],
    evidence: dict[str, Any],
    replay_results: list[dict[str, Any]],
    source_match: dict[str, Any],
    policy_audit: dict[str, Any],
    replay_audit: dict[str, Any],
    canary_audit: dict[str, Any],
    boundary: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    reason_codes = unique_sorted(
        list(evidence["read_reason_codes"])
        + _scope_reason_codes(config, evidence, replay_results, policy_audit)
        + list(source_match["reason_codes"])
        + list(replay_audit["reason_codes"])
        + list(canary_audit["reason_codes"])
        + list(boundary["reason_codes"])
        + list(kill_switch["reason_codes"])
        + list(rollback["reason_codes"])
        + list(telemetry["reason_codes"])
    )
    preflight = evidence["shadow_canary_preflight"]
    if any(
        reason in evidence["read_reason_codes"]
        for reason in ("missing_shadow_canary_preflight_summary", "invalid_shadow_canary_preflight_summary")
    ):
        next_required_change = FIX_PREFLIGHT_NEXT_REQUIRED_CHANGE
    elif preflight.get("status") != "passed" or preflight.get("next_required_change") != "global_99_shadow_canary_replay":
        next_required_change = FIX_PREFLIGHT_NEXT_REQUIRED_CHANGE
    elif any(reason.startswith("missing_multi_map") or reason.startswith("invalid_multi_map") for reason in evidence["read_reason_codes"]):
        next_required_change = FIX_MULTI_MAP_NEXT_REQUIRED_CHANGE
    elif _failed_required_count(replay_results) > 0 or _aggregate_coverage(replay_results) + 1.0e-12 < float(config["target_coverage_rate"]):
        next_required_change = FIX_REPLAY_NEXT_REQUIRED_CHANGE
    elif not source_match["source_match_audit_passed"]:
        next_required_change = FIX_DETERMINISM_NEXT_REQUIRED_CHANGE
    elif int(policy_audit.get("policy_worse_than_baseline_count", 0)) > 0 or int(policy_audit.get("controlled_regression_count", 0)) > 0:
        next_required_change = FIX_POLICY_NEXT_REQUIRED_CHANGE
    elif float(policy_audit["policy_guard_fallback_rate"]) > float(config["max_policy_guard_fallback_rate"]):
        next_required_change = FIX_FALLBACK_NEXT_REQUIRED_CHANGE
    elif not boundary["boundary_audit_passed"]:
        next_required_change = BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE
    elif reason_codes:
        next_required_change = "resolve_global_99_shadow_canary_replay_rejections"
    else:
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    passed = not reason_codes and next_required_change == PASS_NEXT_REQUIRED_CHANGE
    return {
        "status": "passed" if passed else "failed",
        "reason_codes": reason_codes,
        "shadow_replay_passed": passed,
        "offline_canary_replay_passed": passed,
        "next_required_change": next_required_change,
    }


def _scope_reason_codes(
    config: dict[str, Any],
    evidence: dict[str, Any],
    replay_results: list[dict[str, Any]],
    policy_audit: dict[str, Any],
) -> list[str]:
    reasons = []
    preflight = evidence["shadow_canary_preflight"]
    if preflight.get("status") != "passed":
        reasons.append("shadow_canary_preflight_not_passed")
    if preflight.get("shadow_canary_preflight_verdict") != PREFLIGHT_PASS_VERDICT:
        reasons.append("shadow_canary_preflight_verdict_invalid")
    if preflight.get("next_required_change") != "global_99_shadow_canary_replay":
        reasons.append("shadow_canary_preflight_next_required_change_invalid")
    if _failed_required_count(replay_results) > 0:
        reasons.append("shadow_replay_required_scenarios_not_passed")
    if _aggregate_coverage(replay_results) + 1.0e-12 < float(config["target_coverage_rate"]):
        reasons.append("shadow_replay_aggregate_coverage_target_not_met")
    if _min_required_coverage(replay_results) + 1.0e-12 < float(config["target_coverage_rate"]):
        reasons.append("shadow_replay_min_coverage_target_not_met")
    if not policy_audit["policy_loaded"] or not policy_audit["policy_guidance_applied"]:
        reasons.append("policy_guidance_unavailable")
    if int(policy_audit.get("policy_worse_than_baseline_count", 0)) > 0 or int(policy_audit.get("controlled_regression_count", 0)) > 0:
        reasons.append("policy_regression_detected")
    if float(policy_audit["policy_guard_fallback_rate"]) > float(config["max_policy_guard_fallback_rate"]):
        reasons.append("policy_guard_fallback_rate_too_high")
    return reasons


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    source_paths: dict[str, Path],
    evidence: dict[str, Any],
    replay_results: list[dict[str, Any]],
    source_match: dict[str, Any],
    policy_audit: dict[str, Any],
    replay_audit: dict[str, Any],
    canary_audit: dict[str, Any],
    boundary: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    preflight = evidence["shadow_canary_preflight"]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "scenario_results": str(paths["scenario_results"]),
        "family_summary": str(paths["family_summary"]),
        "source_match_audit": str(paths["source_match_audit"]),
        "policy_vs_baseline_audit": str(paths["policy_vs_baseline_audit"]),
        "boundary_audit": str(paths["boundary"]),
        "kill_switch_audit": str(paths["kill_switch"]),
        "rollback_audit": str(paths["rollback"]),
        "telemetry_audit": str(paths["telemetry"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "source_shadow_canary_preflight_status": preflight.get("status", "missing"),
        "source_shadow_canary_preflight_verdict": preflight.get("shadow_canary_preflight_verdict", "missing"),
        "target_coverage_rate": 0.99,
        "scenario_count": len(replay_results),
        "required_scenario_count": len(_required_results(replay_results)),
        "failed_required_scenario_count": _failed_required_count(replay_results),
        "aggregate_achieved_coverage_rate": _aggregate_coverage(replay_results),
        "min_scenario_achieved_coverage_rate": _min_required_coverage(replay_results),
        "source_match_audit_passed": source_match["source_match_audit_passed"],
        "max_replay_coverage_delta": source_match["max_replay_coverage_delta"],
        "max_replay_path_cost_delta_m": source_match["max_replay_path_cost_delta_m"],
        "policy_loaded": policy_audit["policy_loaded"],
        "policy_guidance_applied": policy_audit["policy_guidance_applied"],
        "policy_scored_candidate_count": int(policy_audit.get("policy_scored_candidate_count", 0)),
        "policy_guided_decision_count": int(policy_audit.get("policy_guided_decision_count", 0)),
        "policy_guard_fallback_count": int(policy_audit.get("policy_guard_fallback_count", 0)),
        "policy_guard_fallback_rate": float(policy_audit["policy_guard_fallback_rate"]),
        "policy_better_than_baseline_count": int(policy_audit.get("policy_better_than_baseline_count", 0)),
        "policy_worse_than_baseline_count": int(policy_audit.get("policy_worse_than_baseline_count", 0)),
        "controlled_regression_count": int(policy_audit.get("controlled_regression_count", 0)),
        "shadow_replay_passed": decision["shadow_replay_passed"],
        "offline_canary_replay_passed": decision["offline_canary_replay_passed"],
        "boundary_audit_passed": boundary["boundary_audit_passed"],
        "kill_switch_audit_passed": kill_switch["kill_switch_audit_passed"],
        "rollback_audit_passed": rollback["rollback_audit_passed"],
        "telemetry_audit_passed": telemetry["telemetry_audit_passed"],
        "next_required_change": decision["next_required_change"],
        **BOUNDARY_FIELDS,
        "canary_traffic_fraction": 0.0,
        "source_paths": {key: str(value) for key, value in source_paths.items()},
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _manifest(
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "stage": "Global 99 Shadow Canary Replay v1",
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "summary_status": summary["status"],
        "source_match_audit_passed": summary["source_match_audit_passed"],
        "next_required_change": summary["next_required_change"],
        "boundary": {**dict(BOUNDARY_FIELDS), "canary_traffic_fraction": 0.0},
    }


def _rejection_report(
    generated_at: str,
    decision: dict[str, Any],
    boundary: dict[str, Any],
    source_match: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "failure_reason_code_counts": dict(Counter(decision["reason_codes"])),
        "shadow_canary_replay_boundary_violations": boundary["violations"],
        "source_match_reason_codes": source_match["reason_codes"],
        "next_required_change": decision["next_required_change"],
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    lines = [
        "# Global 99 Shadow Canary Replay v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- aggregate_achieved_coverage_rate: `{summary['aggregate_achieved_coverage_rate']}`",
        f"- min_scenario_achieved_coverage_rate: `{summary['min_scenario_achieved_coverage_rate']}`",
        f"- source_match_audit_passed: `{summary['source_match_audit_passed']}`",
        f"- max_replay_coverage_delta: `{summary['max_replay_coverage_delta']}`",
        f"- max_replay_path_cost_delta_m: `{summary['max_replay_path_cost_delta_m']}`",
        f"- policy_guard_fallback_rate: `{summary['policy_guard_fallback_rate']}`",
        "",
        "## Audits",
        "",
        f"- shadow_replay_passed: `{summary['shadow_replay_passed']}`",
        f"- offline_canary_replay_passed: `{summary['offline_canary_replay_passed']}`",
        f"- boundary_audit_passed: `{summary['boundary_audit_passed']}`",
        f"- kill_switch_audit_passed: `{summary['kill_switch_audit_passed']}`",
        f"- rollback_audit_passed: `{summary['rollback_audit_passed']}`",
        f"- telemetry_audit_passed: `{summary['telemetry_audit_passed']}`",
        f"- rejection_reason_counts: `{rejection_report['failure_reason_code_counts']}`",
        "",
        "This stage is an offline synthetic shadow/canary replay only. It does not publish a checkpoint, replace default policy, connect a real executor, start online canary traffic, run PPO, modify network/action space/default A*, call path-planner, use NPZ/sidecar maps, or claim real-world performance.",
        "",
    ]
    return "\n".join(lines)


def _required_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in results if row.get("generalization_required", True)]


def _failed_required_count(results: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in _required_results(results)
        if row.get("status") != "passed" or row.get("coverage_target_met") is not True
    )


def _aggregate_coverage(results: list[dict[str, Any]]) -> float:
    required = _required_results(results)
    denominator = sum(int(row.get("reachable_safe_cell_count", 0)) for row in required)
    covered = sum(int(row.get("covered_reachable_safe_cell_count", 0)) for row in required)
    return covered / denominator if denominator else 0.0


def _min_required_coverage(results: list[dict[str, Any]]) -> float:
    return min((float(row.get("achieved_coverage_rate", 0.0)) for row in _required_results(results)), default=0.0)


def _bounded_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be numeric")
    numeric = float(value)
    if numeric <= 0.0 or numeric > 1.0:
        raise ConfigError(f"{label} must be > 0 and <= 1")
    return numeric


def _non_negative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be numeric")
    numeric = float(value)
    if numeric < 0.0:
        raise ConfigError(f"{label} must be >= 0")
    return numeric


def _fraction_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be numeric")
    numeric = float(value)
    if numeric < 0.0 or numeric > 1.0:
        raise ConfigError(f"{label} must be >= 0 and <= 1")
    return numeric


def _bool_value(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be boolean")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
