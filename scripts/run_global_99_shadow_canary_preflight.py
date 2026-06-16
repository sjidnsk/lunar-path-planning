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
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json


CONFIG_SCHEMA_VERSION = "global-99-shadow-canary-preflight-config/v1"
SUMMARY_SCHEMA_VERSION = "global-99-shadow-canary-preflight-summary/v1"
MANIFEST_SCHEMA_VERSION = "global-99-shadow-canary-manifest/v1"
SHADOW_REPLAY_AUDIT_SCHEMA_VERSION = "global-99-shadow-replay-audit/v1"
CANARY_ELIGIBILITY_AUDIT_SCHEMA_VERSION = "global-99-canary-eligibility-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "global-99-shadow-canary-boundary-audit/v1"
KILL_SWITCH_AUDIT_SCHEMA_VERSION = "global-99-shadow-canary-kill-switch-audit/v1"
ROLLBACK_AUDIT_SCHEMA_VERSION = "global-99-shadow-canary-rollback-audit/v1"
TELEMETRY_AUDIT_SCHEMA_VERSION = "global-99-shadow-canary-telemetry-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "global-99-shadow-canary-rejection-report/v1"

DEFAULT_CONFIG = "configs/global_99_shadow_canary_preflight_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_global_99_shadow_canary_preflight_v1"

RELEASE_GOVERNANCE_SUMMARY_FILE = "global-99-release-governance-preflight-summary.json"
MULTI_MAP_SUMMARY_FILE = "global-99-multi-map-generalization-summary.json"
POLICY_GUIDED_SUMMARY_FILE = "policy-guided-global-coverage-summary.json"

SUMMARY_FILE = "global-99-shadow-canary-preflight-summary.json"
MANIFEST_FILE = "global-99-shadow-canary-manifest.json"
SHADOW_REPLAY_AUDIT_FILE = "global-99-shadow-replay-audit.json"
CANARY_ELIGIBILITY_AUDIT_FILE = "global-99-canary-eligibility-audit.json"
BOUNDARY_AUDIT_FILE = "global-99-shadow-canary-boundary-audit.json"
KILL_SWITCH_AUDIT_FILE = "global-99-shadow-canary-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "global-99-shadow-canary-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "global-99-shadow-canary-telemetry-audit.json"
REJECTION_REPORT_FILE = "global-99-shadow-canary-rejection-report.json"
REPORT_FILE = "global-99-shadow-canary-preflight-report.md"

RELEASE_PASS_VERDICT = "eligible_for_global_99_shadow_canary_preflight"
PASS_VERDICT = "eligible_for_global_99_shadow_canary_replay"
PASS_NEXT_REQUIRED_CHANGE = "global_99_shadow_canary_replay"
FIX_RELEASE_NEXT_REQUIRED_CHANGE = "fix_global_99_release_governance_preflight"
FIX_MULTI_MAP_NEXT_REQUIRED_CHANGE = "fix_global_99_multi_map_generalization"
FIX_POLICY_NEXT_REQUIRED_CHANGE = "fix_policy_guided_global_coverage"
FIX_FALLBACK_NEXT_REQUIRED_CHANGE = "fix_global_99_shadow_canary_guard_fallback"
BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE = "resolve_global_99_shadow_canary_boundary_rejections"

BOUNDARY_FIELDS = {
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

SHADOW_CANARY_BOUNDARY_FIELDS = tuple(BOUNDARY_FIELDS) + (
    "checkpoint_publication_approved",
    "final_release_approved",
    "relaxes_guard",
    "guard_relaxed",
    "modifies_network_or_action_space",
    "ackermann_feasible_trajectory_claimed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Global 99 Shadow Canary Preflight v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_global_99_shadow_canary_preflight(
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
                "shadow_canary_preflight_verdict": summary["shadow_canary_preflight_verdict"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_global_99_shadow_canary_preflight(
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

    lineage = _lineage_audit(source_paths, evidence)
    boundary = _boundary_audit(config, evidence)
    shadow_replay = _shadow_replay_audit(config, evidence, lineage, boundary)
    canary_eligibility = _canary_eligibility_audit(config, evidence, lineage, boundary)
    kill_switch = _kill_switch_audit(lineage, boundary)
    rollback = _rollback_audit(lineage, boundary)
    telemetry = _telemetry_audit(evidence)
    decision = _decision(
        config,
        evidence,
        lineage,
        shadow_replay,
        canary_eligibility,
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
        decision=decision,
        shadow_replay=shadow_replay,
        canary_eligibility=canary_eligibility,
        boundary=boundary,
        kill_switch=kill_switch,
        rollback=rollback,
        telemetry=telemetry,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)
    rejection_report = _rejection_report(generated_at, decision, boundary)

    for key, payload in (
        ("shadow_replay", shadow_replay),
        ("canary_eligibility", canary_eligibility),
        ("boundary", boundary),
        ("kill_switch", kill_switch),
        ("rollback", rollback),
        ("telemetry", telemetry),
        ("rejection_report", rejection_report),
        ("manifest", manifest),
        ("summary", summary),
    ):
        write_json(paths[key], payload)
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
    for key in ("source_release_governance_root", "source_multi_map_root", "source_policy_guided_root"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    normalized = dict(payload)
    normalized["source_release_governance_root"] = str(
        resolve_path(Path(payload["source_release_governance_root"]), repo_root)
    )
    normalized["source_multi_map_root"] = str(resolve_path(Path(payload["source_multi_map_root"]), repo_root))
    normalized["source_policy_guided_root"] = str(resolve_path(Path(payload["source_policy_guided_root"]), repo_root))
    normalized["target_coverage_rate"] = _bounded_float(payload.get("target_coverage_rate"), "target_coverage_rate")
    normalized["max_policy_guard_fallback_rate"] = _bounded_float(
        payload.get("max_policy_guard_fallback_rate"),
        "max_policy_guard_fallback_rate",
    )
    normalized["shadow_mode"] = _bool_value(payload.get("shadow_mode"), "shadow_mode")
    normalized["canary_mode"] = _bool_value(payload.get("canary_mode"), "canary_mode")
    normalized["canary_traffic_fraction"] = _fraction_float(
        payload.get("canary_traffic_fraction"),
        "canary_traffic_fraction",
    )
    normalized["require_default_policy_authoritative"] = _bool_value(
        payload.get("require_default_policy_authoritative"),
        "require_default_policy_authoritative",
    )
    normalized["require_policy_read_only"] = _bool_value(payload.get("require_policy_read_only"), "require_policy_read_only")
    return normalized


def _source_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    release_root = resolve_path(Path(config["source_release_governance_root"]), repo_root)
    multi_map_root = resolve_path(Path(config["source_multi_map_root"]), repo_root)
    policy_root = resolve_path(Path(config["source_policy_guided_root"]), repo_root)
    return {
        "release_governance_root": release_root,
        "multi_map_root": multi_map_root,
        "policy_guided_root": policy_root,
        "release_governance_summary": release_root / RELEASE_GOVERNANCE_SUMMARY_FILE,
        "multi_map_summary": multi_map_root / MULTI_MAP_SUMMARY_FILE,
        "policy_guided_summary": policy_root / POLICY_GUIDED_SUMMARY_FILE,
    }


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "shadow_replay": output_root / SHADOW_REPLAY_AUDIT_FILE,
        "canary_eligibility": output_root / CANARY_ELIGIBILITY_AUDIT_FILE,
        "boundary": output_root / BOUNDARY_AUDIT_FILE,
        "kill_switch": output_root / KILL_SWITCH_AUDIT_FILE,
        "rollback": output_root / ROLLBACK_AUDIT_FILE,
        "telemetry": output_root / TELEMETRY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _load_evidence(source_paths: dict[str, Path]) -> dict[str, Any]:
    read_reasons: list[str] = []
    release = _read_json(source_paths["release_governance_summary"], read_reasons, "release_governance")
    multi_map = _read_json(source_paths["multi_map_summary"], read_reasons, "multi_map")
    policy_guided = _read_json(source_paths["policy_guided_summary"], read_reasons, "policy_guided")
    return {
        "release_governance": release,
        "multi_map": multi_map,
        "policy_guided": policy_guided,
        "read_reason_codes": unique_sorted(read_reasons),
    }


def _read_json(path: Path, reasons: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(f"missing_{label}_summary")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        reasons.append(f"invalid_{label}_summary")
        return {}
    if not isinstance(payload, dict):
        reasons.append(f"invalid_{label}_summary")
        return {}
    return payload


def _lineage_audit(source_paths: dict[str, Path], evidence: dict[str, Any]) -> dict[str, Any]:
    reasons = list(evidence["read_reason_codes"])
    release = evidence["release_governance"]
    multi_map = evidence["multi_map"]
    policy_guided = evidence["policy_guided"]
    if release and release.get("next_required_change") != "global_99_shadow_canary_preflight":
        reasons.append("release_governance_next_required_change_invalid")
    if release and release.get("release_governance_verdict") != RELEASE_PASS_VERDICT:
        reasons.append("release_governance_verdict_invalid")
    if multi_map and multi_map.get("next_required_change") != "network_architecture_upgrade_readiness_review":
        reasons.append("multi_map_next_required_change_invalid")
    if policy_guided and policy_guided.get("next_required_change") != "global_99_multi_map_generalization":
        reasons.append("policy_guided_next_required_change_invalid")
    return {
        "schema_version": "global-99-shadow-canary-evidence-lineage-audit/v1",
        "evidence_lineage_audit_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "source_paths": {key: str(value) for key, value in source_paths.items()},
    }


def _shadow_replay_audit(
    config: dict[str, Any],
    evidence: dict[str, Any],
    lineage: dict[str, Any],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    release = evidence["release_governance"]
    reasons: list[str] = []
    if not config["shadow_mode"]:
        reasons.append("shadow_mode_disabled")
    if release.get("status") != "passed":
        reasons.append("release_governance_not_passed")
    if not release.get("release_governance_preflight_passed", False):
        reasons.append("release_governance_preflight_not_passed")
    if not lineage["evidence_lineage_audit_passed"]:
        reasons.append("shadow_replay_lineage_blocked")
    if not boundary["boundary_audit_passed"]:
        reasons.append("shadow_replay_boundary_blocked")
    return {
        "schema_version": SHADOW_REPLAY_AUDIT_SCHEMA_VERSION,
        "shadow_replay_audit_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "shadow_mode": config["shadow_mode"],
        "shadow_replay_real_executor_connected": False,
        "shadow_replay_real_world_performance_claimed": False,
    }


def _canary_eligibility_audit(
    config: dict[str, Any],
    evidence: dict[str, Any],
    lineage: dict[str, Any],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    policy_guided = evidence["policy_guided"]
    reasons: list[str] = []
    if not config["canary_mode"]:
        reasons.append("canary_mode_disabled")
    if float(config["canary_traffic_fraction"]) != 0.0:
        reasons.append("canary_traffic_fraction_must_be_zero")
    if not config["require_default_policy_authoritative"]:
        reasons.append("default_policy_authoritative_not_required")
    if config["require_policy_read_only"] and policy_guided.get("policy_read_only") is not True:
        reasons.append("policy_read_only_not_verified")
    if not lineage["evidence_lineage_audit_passed"]:
        reasons.append("canary_lineage_blocked")
    if not boundary["boundary_audit_passed"]:
        reasons.append("canary_boundary_blocked")
    return {
        "schema_version": CANARY_ELIGIBILITY_AUDIT_SCHEMA_VERSION,
        "canary_eligibility_audit_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "canary_mode": config["canary_mode"],
        "canary_traffic_fraction": 0.0,
        "default_policy_authoritative": config["require_default_policy_authoritative"],
        "starts_online_canary": False,
    }


def _boundary_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    violations = []
    for source_name in ("release_governance", "multi_map", "policy_guided"):
        payload = evidence[source_name]
        for field in SHADOW_CANARY_BOUNDARY_FIELDS:
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
    if float(config["canary_traffic_fraction"]) != 0.0:
        violations.append({"source": "config", "field": "canary_traffic_fraction", "value": config["canary_traffic_fraction"]})
    return {
        "schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION,
        "boundary_audit_passed": not violations,
        "reason_codes": [] if not violations else ["shadow_canary_boundary_violation"],
        "violations": violations,
        **BOUNDARY_FIELDS,
        "canary_traffic_fraction": 0.0,
    }


def _kill_switch_audit(lineage: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    passed = lineage["evidence_lineage_audit_passed"] and boundary["boundary_audit_passed"]
    return {
        "schema_version": KILL_SWITCH_AUDIT_SCHEMA_VERSION,
        "kill_switch_audit_passed": passed,
        "reason_codes": [] if passed else ["kill_switch_contract_blocked"],
        "kill_switch_required_before_shadow_canary_replay": True,
        "kill_switch_real_control_plane_connected": False,
    }


def _rollback_audit(lineage: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    passed = lineage["evidence_lineage_audit_passed"] and boundary["boundary_audit_passed"]
    return {
        "schema_version": ROLLBACK_AUDIT_SCHEMA_VERSION,
        "rollback_audit_passed": passed,
        "reason_codes": [] if passed else ["rollback_contract_blocked"],
        "rollback_required_before_shadow_canary_replay": True,
        "rollback_real_control_plane_connected": False,
    }


def _telemetry_audit(evidence: dict[str, Any]) -> dict[str, Any]:
    release = evidence["release_governance"]
    multi_map = evidence["multi_map"]
    policy_guided = evidence["policy_guided"]
    passed = (
        bool(release.get("policy_guidance_applied", False))
        and bool(multi_map.get("policy_guidance_applied", False))
        and bool(policy_guided.get("policy_guidance_applied", False))
        and int(multi_map.get("policy_scored_candidate_count", 0)) > 0
        and int(multi_map.get("policy_guided_decision_count", 0)) > 0
        and int(policy_guided.get("policy_scored_candidate_count", 0)) > 0
        and int(policy_guided.get("policy_guided_decision_count", 0)) > 0
    )
    return {
        "schema_version": TELEMETRY_AUDIT_SCHEMA_VERSION,
        "telemetry_audit_passed": passed,
        "reason_codes": [] if passed else ["telemetry_evidence_incomplete"],
        "tracked_counters": {
            "policy_scored_candidate_count": int(multi_map.get("policy_scored_candidate_count", 0)),
            "policy_guided_decision_count": int(multi_map.get("policy_guided_decision_count", 0)),
            "policy_guard_fallback_count": int(multi_map.get("policy_guard_fallback_count", 0)),
        },
    }


def _decision(
    config: dict[str, Any],
    evidence: dict[str, Any],
    lineage: dict[str, Any],
    shadow_replay: dict[str, Any],
    canary_eligibility: dict[str, Any],
    boundary: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    release = evidence["release_governance"]
    multi_map = evidence["multi_map"]
    reason_codes = unique_sorted(
        list(lineage["reason_codes"])
        + list(shadow_replay["reason_codes"])
        + list(canary_eligibility["reason_codes"])
        + list(boundary["reason_codes"])
        + list(kill_switch["reason_codes"])
        + list(rollback["reason_codes"])
        + list(telemetry["reason_codes"])
        + _scope_reason_codes(config, evidence)
    )
    if any(reason in evidence["read_reason_codes"] for reason in ("missing_release_governance_summary", "invalid_release_governance_summary")):
        next_required_change = FIX_RELEASE_NEXT_REQUIRED_CHANGE
    elif release.get("status") != "passed" or release.get("next_required_change") != "global_99_shadow_canary_preflight":
        next_required_change = FIX_RELEASE_NEXT_REQUIRED_CHANGE
    elif any(reason in evidence["read_reason_codes"] for reason in ("missing_multi_map_summary", "invalid_multi_map_summary")):
        next_required_change = FIX_MULTI_MAP_NEXT_REQUIRED_CHANGE
    elif multi_map.get("status") != "passed":
        next_required_change = FIX_MULTI_MAP_NEXT_REQUIRED_CHANGE
    elif (
        int(multi_map.get("failed_required_scenario_count", 0)) > 0
        or float(multi_map.get("aggregate_achieved_coverage_rate", 0.0)) + 1.0e-12 < float(config["target_coverage_rate"])
        or float(multi_map.get("min_scenario_achieved_coverage_rate", 0.0)) + 1.0e-12 < float(config["target_coverage_rate"])
    ):
        next_required_change = FIX_MULTI_MAP_NEXT_REQUIRED_CHANGE
    elif any(reason in evidence["read_reason_codes"] for reason in ("missing_policy_guided_summary", "invalid_policy_guided_summary")):
        next_required_change = FIX_POLICY_NEXT_REQUIRED_CHANGE
    elif evidence["policy_guided"].get("status") != "passed":
        next_required_change = FIX_POLICY_NEXT_REQUIRED_CHANGE
    elif _policy_worse_count(evidence) > 0 or _controlled_regression_count(evidence) > 0:
        next_required_change = FIX_POLICY_NEXT_REQUIRED_CHANGE
    elif _policy_guard_fallback_rate(evidence) > float(config["max_policy_guard_fallback_rate"]):
        next_required_change = FIX_FALLBACK_NEXT_REQUIRED_CHANGE
    elif not boundary["boundary_audit_passed"]:
        next_required_change = BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE
    elif reason_codes:
        next_required_change = "resolve_global_99_shadow_canary_preflight_rejections"
    else:
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    passed = not reason_codes and next_required_change == PASS_NEXT_REQUIRED_CHANGE
    return {
        "status": "passed" if passed else "failed",
        "reason_codes": reason_codes,
        "shadow_preflight_passed": passed,
        "canary_preflight_passed": passed,
        "shadow_canary_preflight_verdict": PASS_VERDICT if passed else "resolve_global_99_shadow_canary_preflight_rejections",
        "next_required_change": next_required_change,
    }


def _scope_reason_codes(config: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    release = evidence["release_governance"]
    multi_map = evidence["multi_map"]
    policy_guided = evidence["policy_guided"]
    target = float(config["target_coverage_rate"])
    reasons: list[str] = []
    if release.get("status") != "passed":
        reasons.append("release_governance_not_passed")
    if multi_map.get("status") != "passed":
        reasons.append("multi_map_not_passed")
    if policy_guided.get("status") != "passed":
        reasons.append("policy_guided_not_passed")
    if int(multi_map.get("failed_required_scenario_count", 1)) > 0:
        reasons.append("multi_map_required_scenarios_not_passed")
    if float(multi_map.get("aggregate_achieved_coverage_rate", 0.0)) + 1.0e-12 < target:
        reasons.append("multi_map_aggregate_coverage_target_not_met")
    if float(multi_map.get("min_scenario_achieved_coverage_rate", 0.0)) + 1.0e-12 < target:
        reasons.append("multi_map_min_coverage_target_not_met")
    if not release.get("policy_guidance_applied") or not multi_map.get("policy_guidance_applied") or not policy_guided.get("policy_guidance_applied"):
        reasons.append("policy_guidance_not_applied")
    if _policy_worse_count(evidence) > 0 or _controlled_regression_count(evidence) > 0:
        reasons.append("policy_regression_detected")
    if _policy_guard_fallback_rate(evidence) > float(config["max_policy_guard_fallback_rate"]):
        reasons.append("policy_guard_fallback_rate_too_high")
    if config["require_policy_read_only"] and policy_guided.get("policy_read_only") is not True:
        reasons.append("policy_read_only_not_verified")
    return reasons


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    source_paths: dict[str, Path],
    evidence: dict[str, Any],
    decision: dict[str, Any],
    shadow_replay: dict[str, Any],
    canary_eligibility: dict[str, Any],
    boundary: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    release = evidence["release_governance"]
    multi_map = evidence["multi_map"]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "shadow_replay_audit": str(paths["shadow_replay"]),
        "canary_eligibility_audit": str(paths["canary_eligibility"]),
        "boundary_audit": str(paths["boundary"]),
        "kill_switch_audit": str(paths["kill_switch"]),
        "rollback_audit": str(paths["rollback"]),
        "telemetry_audit": str(paths["telemetry"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "source_release_governance_status": release.get("status", "missing"),
        "release_governance_verdict": release.get("release_governance_verdict", "missing"),
        "target_coverage_rate": float(multi_map.get("target_coverage_rate", release.get("target_coverage_rate", 0.99))),
        "aggregate_achieved_coverage_rate": float(multi_map.get("aggregate_achieved_coverage_rate", 0.0)),
        "min_scenario_achieved_coverage_rate": float(multi_map.get("min_scenario_achieved_coverage_rate", 0.0)),
        "required_scenario_count": int(multi_map.get("required_scenario_count", release.get("required_scenario_count", 0))),
        "failed_required_scenario_count": int(
            multi_map.get("failed_required_scenario_count", release.get("failed_required_scenario_count", 0))
        ),
        "policy_guidance_applied": bool(
            release.get("policy_guidance_applied")
            and multi_map.get("policy_guidance_applied")
            and evidence["policy_guided"].get("policy_guidance_applied")
        ),
        "policy_guard_fallback_rate": _policy_guard_fallback_rate(evidence),
        "policy_better_than_baseline_count": int(
            max(
                release.get("policy_better_than_baseline_count", 0),
                multi_map.get("policy_better_than_baseline_count", 0),
            )
        ),
        "policy_worse_than_baseline_count": _policy_worse_count(evidence),
        "controlled_regression_count": _controlled_regression_count(evidence),
        "shadow_preflight_passed": decision["shadow_preflight_passed"],
        "canary_preflight_passed": decision["canary_preflight_passed"],
        "shadow_canary_preflight_verdict": decision["shadow_canary_preflight_verdict"],
        "shadow_replay_audit_passed": shadow_replay["shadow_replay_audit_passed"],
        "canary_eligibility_audit_passed": canary_eligibility["canary_eligibility_audit_passed"],
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
        "stage": "Global 99 Shadow Canary Preflight v1",
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "summary_status": summary["status"],
        "shadow_canary_preflight_verdict": summary["shadow_canary_preflight_verdict"],
        "next_required_change": summary["next_required_change"],
        "boundary": {**dict(BOUNDARY_FIELDS), "canary_traffic_fraction": 0.0},
    }


def _rejection_report(generated_at: str, decision: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "failure_reason_code_counts": dict(Counter(decision["reason_codes"])),
        "shadow_canary_boundary_violations": boundary["violations"],
        "next_required_change": decision["next_required_change"],
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    lines = [
        "# Global 99 Shadow Canary Preflight v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- shadow_canary_preflight_verdict: `{summary['shadow_canary_preflight_verdict']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- aggregate_achieved_coverage_rate: `{summary['aggregate_achieved_coverage_rate']}`",
        f"- min_scenario_achieved_coverage_rate: `{summary['min_scenario_achieved_coverage_rate']}`",
        f"- policy_guard_fallback_rate: `{summary['policy_guard_fallback_rate']}`",
        f"- canary_traffic_fraction: `{summary['canary_traffic_fraction']}`",
        "",
        "## Audits",
        "",
        f"- shadow_replay_audit_passed: `{summary['shadow_replay_audit_passed']}`",
        f"- canary_eligibility_audit_passed: `{summary['canary_eligibility_audit_passed']}`",
        f"- boundary_audit_passed: `{summary['boundary_audit_passed']}`",
        f"- kill_switch_audit_passed: `{summary['kill_switch_audit_passed']}`",
        f"- rollback_audit_passed: `{summary['rollback_audit_passed']}`",
        f"- telemetry_audit_passed: `{summary['telemetry_audit_passed']}`",
        f"- rejection_reason_counts: `{rejection_report['failure_reason_code_counts']}`",
        "",
        "This stage is an offline shadow/canary preflight only. It does not publish a checkpoint, replace default policy, connect a real executor, start online canary traffic, run PPO, modify network/action space/default A*, call path-planner, use NPZ/sidecar maps, or claim real-world performance.",
        "",
    ]
    return "\n".join(lines)


def _policy_guard_fallback_rate(evidence: dict[str, Any]) -> float:
    release = evidence["release_governance"]
    multi_map = evidence["multi_map"]
    return float(max(release.get("policy_guard_fallback_rate", 0.0), multi_map.get("policy_guard_fallback_rate", 0.0)))


def _policy_worse_count(evidence: dict[str, Any]) -> int:
    return int(
        max(
            evidence["release_governance"].get("policy_worse_than_baseline_count", 0),
            evidence["multi_map"].get("policy_worse_than_baseline_count", 0),
            evidence["policy_guided"].get("policy_worse_than_baseline_count", 0),
        )
    )


def _controlled_regression_count(evidence: dict[str, Any]) -> int:
    return int(
        max(
            evidence["release_governance"].get("controlled_regression_count", 0),
            evidence["multi_map"].get("controlled_regression_count", 0),
            evidence["policy_guided"].get("controlled_regression_count", 0),
        )
    )


def _bounded_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be numeric")
    numeric = float(value)
    if numeric <= 0.0 or numeric > 1.0:
        raise ConfigError(f"{label} must be > 0 and <= 1")
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
