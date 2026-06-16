from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json
    from global_99_governance_common import global_99_boundary_defaults
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json
    from scripts.global_99_governance_common import global_99_boundary_defaults


CONFIG_SCHEMA_VERSION = "xunce-release-governance-gate-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-release-governance-gate-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-release-governance-gate-manifest/v1"
LINEAGE_AUDIT_SCHEMA_VERSION = "xunce-release-evidence-lineage-audit/v1"
SCOPE_AUDIT_SCHEMA_VERSION = "xunce-release-scope-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-release-boundary-audit/v1"
DECISION_AUDIT_SCHEMA_VERSION = "xunce-release-governance-decision-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-release-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_release_governance_gate_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_release_governance_gate_v1"

SUMMARY_FILE = "xunce-release-governance-gate-summary.json"
MANIFEST_FILE = "xunce-release-governance-gate-manifest.json"
LINEAGE_AUDIT_FILE = "xunce-release-evidence-lineage-audit.json"
SCOPE_AUDIT_FILE = "xunce-release-scope-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-release-boundary-audit.json"
DECISION_AUDIT_FILE = "xunce-release-governance-decision-audit.json"
REJECTION_REPORT_FILE = "xunce-release-rejection-report.json"
REPORT_FILE = "xunce-release-governance-gate-report.md"

SOURCE_SANDBOX_SUMMARY_FILE = "xunce-sandbox-candidate-preflight-summary.json"
SOURCE_SANDBOX_PASS_NEXT_REQUIRED_CHANGE = "xunce_release_governance_gate"

PASS_NEXT_REQUIRED_CHANGE = "xunce_research_track_complete"
FIX_SANDBOX_NEXT_REQUIRED_CHANGE = "fix_xunce_sandbox_candidate_preflight"
FAIL_NEXT_REQUIRED_CHANGE = "fix_xunce_release_governance_gate"
BOUNDARY_NEXT_REQUIRED_CHANGE = "resolve_xunce_release_governance_boundary_rejections"

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "real_world_release_approved",
    "real_world_performance_claimed",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "starts_online_canary",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce Release Governance Gate v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_release_governance_gate(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"], "next_required_change": summary["next_required_change"], "summary": summary["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_release_governance_gate(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    sandbox_root = resolve_path(Path(config["source_sandbox_candidate_root"]), repo_root)
    sandbox_summary = _load_json(sandbox_root / SOURCE_SANDBOX_SUMMARY_FILE)
    lineage_audit = _lineage_audit(sandbox_summary)
    scope_audit = _scope_audit(sandbox_summary, config)
    boundary_audit = _boundary_audit(sandbox_summary)
    decision_audit = _decision_audit(lineage_audit, scope_audit, boundary_audit)
    decision = _decision(lineage_audit, scope_audit, boundary_audit, decision_audit)
    generated_at = utc_now()
    summary = _summary(generated_at, config_path, output_root, paths, sandbox_summary, lineage_audit, scope_audit, boundary_audit, decision_audit, decision, repo_root)
    manifest = {"schema_version": MANIFEST_SCHEMA_VERSION, "generated_at": generated_at, "config": str(config_path), "output_root": str(output_root), "artifacts": {key: str(path) for key, path in paths.items()}, "summary_status": summary["status"], "next_required_change": summary["next_required_change"]}
    rejection_report = {"schema_version": REJECTION_REPORT_SCHEMA_VERSION, "status": decision["status"], "reason_codes": decision["reason_codes"], "next_required_change": decision["next_required_change"], "release_governance_gate_passed": summary["release_governance_gate_passed"]}
    write_json(paths["lineage_audit"], lineage_audit)
    write_json(paths["scope_audit"], scope_audit)
    write_json(paths["boundary_audit"], boundary_audit)
    write_json(paths["decision_audit"], decision_audit)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _load_config(path: Path) -> dict[str, Any]:
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
    if not isinstance(payload.get("source_sandbox_candidate_root"), str) or not payload["source_sandbox_candidate_root"].strip():
        raise ConfigError("source_sandbox_candidate_root must be a non-empty string")
    normalized = dict(payload)
    for key in ("require_sandbox_preflight_passed", "require_kill_switch", "require_rollback", "require_telemetry", "require_default_policy_read_only", "require_executor_isolation"):
        normalized[key] = bool(normalized.get(key, True))
    return normalized


def _lineage_audit(sandbox_summary: dict[str, Any] | None) -> dict[str, Any]:
    reason_codes: list[str] = []
    if not isinstance(sandbox_summary, dict):
        reason_codes.append("missing_sandbox_candidate_preflight_summary")
    else:
        if sandbox_summary.get("status") != "passed":
            reason_codes.append("sandbox_candidate_preflight_not_passed")
        if sandbox_summary.get("next_required_change") != SOURCE_SANDBOX_PASS_NEXT_REQUIRED_CHANGE:
            reason_codes.append("sandbox_candidate_wrong_next_required_change")
        if sandbox_summary.get("sandbox_candidate_preflight_passed") is not True:
            reason_codes.append("sandbox_candidate_pass_field_false")
    return {"schema_version": LINEAGE_AUDIT_SCHEMA_VERSION, "source_status": sandbox_summary.get("status") if isinstance(sandbox_summary, dict) else None, "source_next_required_change": sandbox_summary.get("next_required_change") if isinstance(sandbox_summary, dict) else None, "reason_codes": unique_sorted(reason_codes), "evidence_lineage_audit_passed": not reason_codes}


def _scope_audit(sandbox_summary: dict[str, Any] | None, config: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    checks = {
        "sandbox_package_created": "sandbox_package_not_created",
        "sandbox_load_verified": "sandbox_load_not_verified",
        "kill_switch_audit_passed": "sandbox_kill_switch_not_passed",
        "rollback_audit_passed": "sandbox_rollback_not_passed",
        "telemetry_audit_passed": "sandbox_telemetry_not_passed",
        "default_policy_read_only": "default_policy_not_read_only",
        "executor_isolation_passed": "executor_isolation_not_passed",
    }
    if isinstance(sandbox_summary, dict):
        for field, reason in checks.items():
            if sandbox_summary.get(field) is not True:
                reason_codes.append(reason)
    elif config["require_sandbox_preflight_passed"]:
        reason_codes.append("sandbox_scope_unavailable")
    return {"schema_version": SCOPE_AUDIT_SCHEMA_VERSION, "reason_codes": unique_sorted(reason_codes), "scope_audit_passed": not reason_codes, **{field: bool(sandbox_summary.get(field, False)) if isinstance(sandbox_summary, dict) else False for field in checks}}


def _boundary_audit(sandbox_summary: dict[str, Any] | None) -> dict[str, Any]:
    violations: list[str] = []
    observed: dict[str, bool] = {}
    if isinstance(sandbox_summary, dict):
        for field in BOUNDARY_FIELDS:
            value = bool(sandbox_summary.get(field, False))
            observed[field] = value
            if value:
                violations.append(field)
    return {"schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION, "observed_source_boundary_fields": observed, "violating_source_boundary_fields": unique_sorted(violations), "boundary_audit_passed": not violations, **_closed_boundary_fields()}


def _decision_audit(lineage_audit: dict[str, Any], scope_audit: dict[str, Any], boundary_audit: dict[str, Any]) -> dict[str, Any]:
    passed = lineage_audit["evidence_lineage_audit_passed"] and scope_audit["scope_audit_passed"] and boundary_audit["boundary_audit_passed"]
    return {"schema_version": DECISION_AUDIT_SCHEMA_VERSION, "release_governance_gate_passed": passed, "release_governance_verdict": "research_candidate_ready_for_human_governance_review" if passed else "blocked", "xunce_research_chain_complete": passed, "default_policy_replacement_approved": False, "real_world_release_approved": False}


def _decision(lineage_audit: dict[str, Any], scope_audit: dict[str, Any], boundary_audit: dict[str, Any], decision_audit: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    reason_codes.extend(lineage_audit["reason_codes"])
    reason_codes.extend(scope_audit["reason_codes"])
    if not boundary_audit["boundary_audit_passed"]:
        reason_codes.append("release_governance_boundary_violation")
    reason_codes = unique_sorted(reason_codes)
    if not reason_codes:
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif "release_governance_boundary_violation" in reason_codes:
        next_required_change = BOUNDARY_NEXT_REQUIRED_CHANGE
    elif any(code.startswith("missing_sandbox") or code.startswith("sandbox_candidate") for code in reason_codes):
        next_required_change = FIX_SANDBOX_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FAIL_NEXT_REQUIRED_CHANGE
    return {"status": "passed" if not reason_codes else "failed", "reason_codes": reason_codes, "next_required_change": next_required_change}


def _summary(generated_at: str, config_path: Path, output_root: Path, paths: dict[str, Path], sandbox_summary: dict[str, Any] | None, lineage_audit: dict[str, Any], scope_audit: dict[str, Any], boundary_audit: dict[str, Any], decision_audit: dict[str, Any], decision: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    summary = {"schema_version": SUMMARY_SCHEMA_VERSION, "generated_at": generated_at, "config": str(config_path), "output_root": str(output_root), "summary": str(paths["summary"]), "status": decision["status"], "reason_codes": decision["reason_codes"], "source_sandbox_status": sandbox_summary.get("status") if isinstance(sandbox_summary, dict) else None, "source_sandbox_next_required_change": sandbox_summary.get("next_required_change") if isinstance(sandbox_summary, dict) else None, "evidence_lineage_audit_passed": lineage_audit["evidence_lineage_audit_passed"], "scope_audit_passed": scope_audit["scope_audit_passed"], "boundary_audit_passed": boundary_audit["boundary_audit_passed"], "release_governance_gate_passed": decision_audit["release_governance_gate_passed"] and decision["status"] == "passed", "release_governance_verdict": decision_audit["release_governance_verdict"] if decision["status"] == "passed" else "blocked", "xunce_research_chain_complete": decision_audit["xunce_research_chain_complete"] and decision["status"] == "passed", "next_required_change": decision["next_required_change"], "git_provenance": {"current": git_snapshot(repo_root)}}
    summary.update(_closed_boundary_fields())
    return summary


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {"summary": output_root / SUMMARY_FILE, "manifest": output_root / MANIFEST_FILE, "lineage_audit": output_root / LINEAGE_AUDIT_FILE, "scope_audit": output_root / SCOPE_AUDIT_FILE, "boundary_audit": output_root / BOUNDARY_AUDIT_FILE, "decision_audit": output_root / DECISION_AUDIT_FILE, "rejection_report": output_root / REJECTION_REPORT_FILE, "report": output_root / REPORT_FILE}


def _closed_boundary_fields() -> dict[str, bool | float]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields.update({"real_world_release_approved": False, "real_world_performance_claimed": False, "default_policy_replacement_approved": False, "real_executor_connection_approved": False, "starts_online_canary": False, "canary_traffic_fraction": 0.0, "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False, "runs_new_training_update": False, "runs_new_ppo_update": False, "modifies_network": False, "modifies_action_space": False, "modifies_default_astar": False})
    return fields


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(["# Xunce Release Governance Gate v1", "", f"- status: `{summary['status']}`", f"- reason_codes: `{summary['reason_codes']}`", f"- release_governance_verdict: `{summary['release_governance_verdict']}`", f"- xunce_research_chain_complete: `{summary['xunce_research_chain_complete']}`", f"- default_policy_replacement_approved: `{summary['default_policy_replacement_approved']}`", f"- real_world_release_approved: `{summary['real_world_release_approved']}`", f"- next_required_change: `{summary['next_required_change']}`", "", "This gate completes the research evidence chain only. It does not approve default-policy replacement, real-world release, executor connection, or online canary traffic.", ""])


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())
