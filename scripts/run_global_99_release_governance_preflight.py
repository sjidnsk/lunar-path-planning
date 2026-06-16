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


CONFIG_SCHEMA_VERSION = "global-99-release-governance-preflight-config/v1"
SUMMARY_SCHEMA_VERSION = "global-99-release-governance-preflight-summary/v1"
MANIFEST_SCHEMA_VERSION = "global-99-release-governance-manifest/v1"
LINEAGE_AUDIT_SCHEMA_VERSION = "global-99-release-evidence-lineage-audit/v1"
SCOPE_AUDIT_SCHEMA_VERSION = "global-99-release-scope-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "global-99-release-boundary-audit/v1"
KILL_SWITCH_AUDIT_SCHEMA_VERSION = "global-99-release-kill-switch-audit/v1"
ROLLBACK_AUDIT_SCHEMA_VERSION = "global-99-release-rollback-audit/v1"
TELEMETRY_AUDIT_SCHEMA_VERSION = "global-99-release-telemetry-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "global-99-release-rejection-report/v1"

DEFAULT_CONFIG = "configs/global_99_release_governance_preflight_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_global_99_release_governance_preflight_v1"

NETWORK_SUMMARY_FILE = "network-architecture-upgrade-readiness-summary.json"
MULTI_MAP_SUMMARY_FILE = "global-99-multi-map-generalization-summary.json"
POLICY_GUIDED_SUMMARY_FILE = "policy-guided-global-coverage-summary.json"

SUMMARY_FILE = "global-99-release-governance-preflight-summary.json"
MANIFEST_FILE = "global-99-release-governance-manifest.json"
LINEAGE_AUDIT_FILE = "global-99-release-evidence-lineage-audit.json"
SCOPE_AUDIT_FILE = "global-99-release-scope-audit.json"
BOUNDARY_AUDIT_FILE = "global-99-release-boundary-audit.json"
KILL_SWITCH_AUDIT_FILE = "global-99-release-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "global-99-release-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "global-99-release-telemetry-audit.json"
REJECTION_REPORT_FILE = "global-99-release-rejection-report.json"
REPORT_FILE = "global-99-release-governance-preflight-report.md"

PASS_VERDICT = "eligible_for_global_99_shadow_canary_preflight"
PASS_NEXT_REQUIRED_CHANGE = "global_99_shadow_canary_preflight"
FIX_NETWORK_NEXT_REQUIRED_CHANGE = "fix_network_architecture_upgrade_readiness_review"
FIX_MULTI_MAP_NEXT_REQUIRED_CHANGE = "fix_global_99_multi_map_generalization"
NETWORK_UPGRADE_NEXT_REQUIRED_CHANGE = "network_architecture_upgrade_v1"
FIX_POLICY_NEXT_REQUIRED_CHANGE = "fix_policy_guided_global_coverage"
BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE = "resolve_global_99_release_boundary_rejections"

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
}

RELEASE_BOUNDARY_FIELDS = tuple(BOUNDARY_FIELDS) + (
    "checkpoint_publication_approved",
    "final_release_approved",
    "relaxes_guard",
    "guard_relaxed",
    "modifies_network_or_action_space",
    "ackermann_feasible_trajectory_claimed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Global 99 Release Governance Preflight v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_global_99_release_governance_preflight(
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
                "release_governance_verdict": summary["release_governance_verdict"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_global_99_release_governance_preflight(
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
    scope = _scope_audit(config, evidence)
    boundary = _boundary_audit(evidence)
    kill_switch = _kill_switch_audit(lineage, boundary)
    rollback = _rollback_audit(lineage, boundary)
    telemetry = _telemetry_audit(evidence)
    decision = _decision(config, evidence, lineage, scope, boundary, kill_switch, rollback, telemetry)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        source_paths=source_paths,
        evidence=evidence,
        decision=decision,
        lineage=lineage,
        scope=scope,
        boundary=boundary,
        kill_switch=kill_switch,
        rollback=rollback,
        telemetry=telemetry,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)
    rejection_report = _rejection_report(generated_at, decision, boundary)

    for key, payload in (
        ("lineage", lineage),
        ("scope", scope),
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
    for key in ("source_network_readiness_root", "source_multi_map_root", "source_policy_guided_root"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    normalized = dict(payload)
    normalized["source_network_readiness_root"] = str(resolve_path(Path(payload["source_network_readiness_root"]), repo_root))
    normalized["source_multi_map_root"] = str(resolve_path(Path(payload["source_multi_map_root"]), repo_root))
    normalized["source_policy_guided_root"] = str(resolve_path(Path(payload["source_policy_guided_root"]), repo_root))
    normalized["target_coverage_rate"] = _bounded_float(payload.get("target_coverage_rate"), "target_coverage_rate")
    normalized["max_policy_guard_fallback_rate"] = _bounded_float(
        payload.get("max_policy_guard_fallback_rate"),
        "max_policy_guard_fallback_rate",
    )
    normalized["require_network_upgrade_recommended_false"] = _bool_value(
        payload.get("require_network_upgrade_recommended_false"),
        "require_network_upgrade_recommended_false",
    )
    normalized["require_required_scenarios_all_passed"] = _bool_value(
        payload.get("require_required_scenarios_all_passed"),
        "require_required_scenarios_all_passed",
    )
    normalized["require_policy_read_only"] = _bool_value(payload.get("require_policy_read_only"), "require_policy_read_only")
    return normalized


def _source_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    network_root = resolve_path(Path(config["source_network_readiness_root"]), repo_root)
    multi_map_root = resolve_path(Path(config["source_multi_map_root"]), repo_root)
    policy_root = resolve_path(Path(config["source_policy_guided_root"]), repo_root)
    return {
        "network_readiness_root": network_root,
        "multi_map_root": multi_map_root,
        "policy_guided_root": policy_root,
        "network_readiness_summary": network_root / NETWORK_SUMMARY_FILE,
        "multi_map_summary": multi_map_root / MULTI_MAP_SUMMARY_FILE,
        "policy_guided_summary": policy_root / POLICY_GUIDED_SUMMARY_FILE,
    }


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "lineage": output_root / LINEAGE_AUDIT_FILE,
        "scope": output_root / SCOPE_AUDIT_FILE,
        "boundary": output_root / BOUNDARY_AUDIT_FILE,
        "kill_switch": output_root / KILL_SWITCH_AUDIT_FILE,
        "rollback": output_root / ROLLBACK_AUDIT_FILE,
        "telemetry": output_root / TELEMETRY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _load_evidence(source_paths: dict[str, Path]) -> dict[str, Any]:
    read_reasons: list[str] = []
    network = _read_json(source_paths["network_readiness_summary"], read_reasons, "network_readiness")
    multi_map = _read_json(source_paths["multi_map_summary"], read_reasons, "multi_map")
    policy_guided = _read_json(source_paths["policy_guided_summary"], read_reasons, "policy_guided")
    return {
        "network_readiness": network,
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
    network = evidence["network_readiness"]
    multi_map = evidence["multi_map"]
    policy_guided = evidence["policy_guided"]
    if network and network.get("source_multi_map_status") != "passed":
        reasons.append("network_readiness_source_multi_map_not_passed")
    if network and network.get("next_required_change") != "global_99_release_governance_preflight":
        reasons.append("network_readiness_next_required_change_invalid")
    if multi_map and multi_map.get("next_required_change") != "network_architecture_upgrade_readiness_review":
        reasons.append("multi_map_next_required_change_invalid")
    if policy_guided and policy_guided.get("next_required_change") != "global_99_multi_map_generalization":
        reasons.append("policy_guided_next_required_change_invalid")
    return {
        "schema_version": LINEAGE_AUDIT_SCHEMA_VERSION,
        "evidence_lineage_audit_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "source_paths": {key: str(value) for key, value in source_paths.items()},
    }


def _scope_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    network = evidence["network_readiness"]
    multi_map = evidence["multi_map"]
    policy_guided = evidence["policy_guided"]
    target = float(config["target_coverage_rate"])
    reasons: list[str] = []
    if network.get("status") != "passed":
        reasons.append("network_readiness_not_passed")
    if multi_map.get("status") != "passed":
        reasons.append("multi_map_not_passed")
    if policy_guided.get("status") != "passed":
        reasons.append("policy_guided_not_passed")
    if config["require_required_scenarios_all_passed"] and int(multi_map.get("failed_required_scenario_count", 1)) > 0:
        reasons.append("multi_map_required_scenarios_not_passed")
    if float(multi_map.get("aggregate_achieved_coverage_rate", 0.0)) + 1.0e-12 < target:
        reasons.append("multi_map_aggregate_coverage_target_not_met")
    if float(multi_map.get("min_scenario_achieved_coverage_rate", 0.0)) + 1.0e-12 < target:
        reasons.append("multi_map_min_coverage_target_not_met")
    if network.get("network_upgrade_recommended") is True and config["require_network_upgrade_recommended_false"]:
        reasons.append("network_upgrade_recommended")
    if config["require_policy_read_only"] and policy_guided.get("policy_read_only") is not True:
        reasons.append("policy_read_only_not_verified")
    if not network.get("policy_guidance_applied") or not multi_map.get("policy_guidance_applied") or not policy_guided.get("policy_guidance_applied"):
        reasons.append("policy_guidance_not_applied")
    if float(network.get("policy_guard_fallback_rate", 1.0)) > float(config["max_policy_guard_fallback_rate"]):
        reasons.append("policy_guard_fallback_rate_too_high")
    if int(network.get("policy_worse_than_baseline_count", 0)) > 0 or int(network.get("controlled_regression_count", 0)) > 0:
        reasons.append("policy_regression_detected")
    return {
        "schema_version": SCOPE_AUDIT_SCHEMA_VERSION,
        "scope_audit_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "target_coverage_rate": target,
        "max_policy_guard_fallback_rate": config["max_policy_guard_fallback_rate"],
    }


def _boundary_audit(evidence: dict[str, Any]) -> dict[str, Any]:
    violations = []
    for source_name in ("network_readiness", "multi_map", "policy_guided"):
        payload = evidence[source_name]
        for field in RELEASE_BOUNDARY_FIELDS:
            if payload.get(field) is True:
                violations.append({"source": source_name, "field": field, "value": True})
    return {
        "schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION,
        "release_boundary_audit_passed": not violations,
        "reason_codes": [] if not violations else ["release_boundary_violation"],
        "violations": violations,
        **BOUNDARY_FIELDS,
    }


def _kill_switch_audit(lineage: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    passed = lineage["evidence_lineage_audit_passed"] and boundary["release_boundary_audit_passed"]
    return {
        "schema_version": KILL_SWITCH_AUDIT_SCHEMA_VERSION,
        "kill_switch_audit_passed": passed,
        "reason_codes": [] if passed else ["kill_switch_contract_blocked"],
        "kill_switch_required_before_shadow_canary": True,
        "kill_switch_real_control_plane_connected": False,
    }


def _rollback_audit(lineage: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    passed = lineage["evidence_lineage_audit_passed"] and boundary["release_boundary_audit_passed"]
    return {
        "schema_version": ROLLBACK_AUDIT_SCHEMA_VERSION,
        "rollback_audit_passed": passed,
        "reason_codes": [] if passed else ["rollback_contract_blocked"],
        "rollback_required_before_shadow_canary": True,
        "rollback_real_control_plane_connected": False,
    }


def _telemetry_audit(evidence: dict[str, Any]) -> dict[str, Any]:
    network = evidence["network_readiness"]
    multi_map = evidence["multi_map"]
    policy_guided = evidence["policy_guided"]
    passed = (
        int(network.get("policy_scored_candidate_count", 0)) > 0
        and int(network.get("policy_guided_decision_count", 0)) > 0
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
            "policy_scored_candidate_count": int(network.get("policy_scored_candidate_count", 0)),
            "policy_guided_decision_count": int(network.get("policy_guided_decision_count", 0)),
            "policy_guard_fallback_count": int(network.get("policy_guard_fallback_count", 0)),
        },
    }


def _decision(
    config: dict[str, Any],
    evidence: dict[str, Any],
    lineage: dict[str, Any],
    scope: dict[str, Any],
    boundary: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    network = evidence["network_readiness"]
    multi_map = evidence["multi_map"]
    reason_codes = unique_sorted(
        list(lineage["reason_codes"])
        + list(scope["reason_codes"])
        + list(boundary["reason_codes"])
        + list(kill_switch["reason_codes"])
        + list(rollback["reason_codes"])
        + list(telemetry["reason_codes"])
    )
    if evidence["read_reason_codes"]:
        next_required_change = FIX_NETWORK_NEXT_REQUIRED_CHANGE if "missing_network_readiness_summary" in evidence["read_reason_codes"] else FIX_MULTI_MAP_NEXT_REQUIRED_CHANGE
    elif network.get("status") != "passed" or network.get("next_required_change") != "global_99_release_governance_preflight":
        next_required_change = FIX_NETWORK_NEXT_REQUIRED_CHANGE
    elif network.get("network_upgrade_recommended") is True and config["require_network_upgrade_recommended_false"]:
        next_required_change = NETWORK_UPGRADE_NEXT_REQUIRED_CHANGE
    elif (
        int(multi_map.get("failed_required_scenario_count", 0)) > 0
        or float(multi_map.get("aggregate_achieved_coverage_rate", 0.0)) + 1.0e-12 < float(config["target_coverage_rate"])
        or float(multi_map.get("min_scenario_achieved_coverage_rate", 0.0)) + 1.0e-12 < float(config["target_coverage_rate"])
    ):
        next_required_change = FIX_MULTI_MAP_NEXT_REQUIRED_CHANGE
    elif int(network.get("policy_worse_than_baseline_count", 0)) > 0 or int(network.get("controlled_regression_count", 0)) > 0:
        next_required_change = FIX_POLICY_NEXT_REQUIRED_CHANGE
    elif not boundary["release_boundary_audit_passed"]:
        next_required_change = BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE
    elif reason_codes:
        next_required_change = "resolve_global_99_release_governance_preflight_rejections"
    else:
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    passed = not reason_codes and next_required_change == PASS_NEXT_REQUIRED_CHANGE
    return {
        "status": "passed" if passed else "failed",
        "reason_codes": reason_codes,
        "release_governance_preflight_passed": passed,
        "release_governance_verdict": PASS_VERDICT if passed else "resolve_global_99_release_governance_preflight_rejections",
        "next_required_change": next_required_change,
    }


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    source_paths: dict[str, Path],
    evidence: dict[str, Any],
    decision: dict[str, Any],
    lineage: dict[str, Any],
    scope: dict[str, Any],
    boundary: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    network = evidence["network_readiness"]
    multi_map = evidence["multi_map"]
    policy_guided = evidence["policy_guided"]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "evidence_lineage_audit": str(paths["lineage"]),
        "scope_audit": str(paths["scope"]),
        "release_boundary_audit": str(paths["boundary"]),
        "kill_switch_audit": str(paths["kill_switch"]),
        "rollback_audit": str(paths["rollback"]),
        "telemetry_audit": str(paths["telemetry"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "source_network_readiness_status": network.get("status", "missing"),
        "source_multi_map_status": multi_map.get("status", "missing"),
        "source_policy_guided_status": policy_guided.get("status", "missing"),
        "target_coverage_rate": float(multi_map.get("target_coverage_rate", network.get("target_coverage_rate", 0.99))),
        "aggregate_achieved_coverage_rate": float(multi_map.get("aggregate_achieved_coverage_rate", 0.0)),
        "min_scenario_achieved_coverage_rate": float(multi_map.get("min_scenario_achieved_coverage_rate", 0.0)),
        "required_scenario_count": int(multi_map.get("required_scenario_count", network.get("required_scenario_count", 0))),
        "failed_required_scenario_count": int(multi_map.get("failed_required_scenario_count", network.get("failed_required_scenario_count", 0))),
        "policy_guidance_applied": bool(network.get("policy_guidance_applied") and multi_map.get("policy_guidance_applied") and policy_guided.get("policy_guidance_applied")),
        "policy_guard_fallback_rate": float(network.get("policy_guard_fallback_rate", 0.0)),
        "policy_better_than_baseline_count": int(network.get("policy_better_than_baseline_count", 0)),
        "policy_worse_than_baseline_count": int(network.get("policy_worse_than_baseline_count", 0)),
        "controlled_regression_count": int(network.get("controlled_regression_count", 0)),
        "network_upgrade_recommended": bool(network.get("network_upgrade_recommended", False)),
        "release_governance_preflight_passed": decision["release_governance_preflight_passed"],
        "release_governance_verdict": decision["release_governance_verdict"],
        "evidence_lineage_audit_passed": lineage["evidence_lineage_audit_passed"],
        "scope_audit_passed": scope["scope_audit_passed"],
        "release_boundary_audit_passed": boundary["release_boundary_audit_passed"],
        "kill_switch_audit_passed": kill_switch["kill_switch_audit_passed"],
        "rollback_audit_passed": rollback["rollback_audit_passed"],
        "telemetry_audit_passed": telemetry["telemetry_audit_passed"],
        "next_required_change": decision["next_required_change"],
        **BOUNDARY_FIELDS,
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
        "stage": "Global 99 Release Governance Preflight v1",
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "summary_status": summary["status"],
        "release_governance_verdict": summary["release_governance_verdict"],
        "next_required_change": summary["next_required_change"],
        "boundary": dict(BOUNDARY_FIELDS),
    }


def _rejection_report(generated_at: str, decision: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "failure_reason_code_counts": dict(Counter(decision["reason_codes"])),
        "release_boundary_violations": boundary["violations"],
        "next_required_change": decision["next_required_change"],
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    lines = [
        "# Global 99 Release Governance Preflight v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- release_governance_verdict: `{summary['release_governance_verdict']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- aggregate_achieved_coverage_rate: `{summary['aggregate_achieved_coverage_rate']}`",
        f"- min_scenario_achieved_coverage_rate: `{summary['min_scenario_achieved_coverage_rate']}`",
        f"- policy_guard_fallback_rate: `{summary['policy_guard_fallback_rate']}`",
        f"- network_upgrade_recommended: `{summary['network_upgrade_recommended']}`",
        "",
        "## Audits",
        "",
        f"- evidence_lineage_audit_passed: `{summary['evidence_lineage_audit_passed']}`",
        f"- scope_audit_passed: `{summary['scope_audit_passed']}`",
        f"- release_boundary_audit_passed: `{summary['release_boundary_audit_passed']}`",
        f"- kill_switch_audit_passed: `{summary['kill_switch_audit_passed']}`",
        f"- rollback_audit_passed: `{summary['rollback_audit_passed']}`",
        f"- telemetry_audit_passed: `{summary['telemetry_audit_passed']}`",
        f"- rejection_reason_counts: `{rejection_report['failure_reason_code_counts']}`",
        "",
        "This stage is a governance preflight only. It does not publish a checkpoint, replace default policy, connect a real executor, run PPO, modify network/action space/default A*, call path-planner, or use NPZ/sidecar maps.",
        "",
    ]
    return "\n".join(lines)


def _bounded_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be numeric")
    numeric = float(value)
    if numeric <= 0.0 or numeric > 1.0:
        raise ConfigError(f"{label} must be > 0 and <= 1")
    return numeric


def _bool_value(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be boolean")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
