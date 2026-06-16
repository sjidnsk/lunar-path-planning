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
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json


CONFIG_SCHEMA_VERSION = "global-99-controlled-default-policy-candidate-installation-preflight-config/v1"
SUMMARY_SCHEMA_VERSION = "global-99-controlled-default-policy-candidate-installation-preflight-summary/v1"
DEFAULT_CONFIG = "configs/global_99_controlled_default_policy_candidate_installation_preflight_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_global_99_controlled_default_policy_candidate_installation_preflight_v1"

CONSUMER_SUMMARY_FILE = "global-99-sandbox-consumer-replay-canary-summary.json"
SUMMARY_FILE = "global-99-controlled-default-policy-candidate-installation-preflight-summary.json"
MANIFEST_FILE = "global-99-controlled-default-policy-candidate-installation-manifest.json"
LINEAGE_AUDIT_FILE = "global-99-controlled-installation-lineage-audit.json"
SCOPE_AUDIT_FILE = "global-99-controlled-installation-review-scope-audit.json"
BOUNDARY_AUDIT_FILE = "global-99-controlled-installation-boundary-audit.json"
KILL_SWITCH_AUDIT_FILE = "global-99-controlled-installation-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "global-99-controlled-installation-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "global-99-controlled-installation-telemetry-audit.json"
REJECTION_REPORT_FILE = "global-99-controlled-installation-rejection-report.json"
REPORT_FILE = "global-99-controlled-default-policy-candidate-installation-preflight-report.md"

PASS_VERDICT = "eligible_for_controlled_default_policy_candidate_installation_review"
PASS_NEXT_REQUIRED_CHANGE = "eligible_for_controlled_default_policy_candidate_installation_review"
FIX_CONSUMER_NEXT_REQUIRED_CHANGE = "fix_sandbox_consumer_replay_canary"
REJECTION_NEXT_REQUIRED_CHANGE = "resolve_controlled_default_policy_candidate_installation_preflight_rejections"

BOUNDARY_FIELDS = {
    "controlled_installation_preflight_only": True,
    "controlled_installation_executed": False,
    "starts_online_canary": False,
    "canary_traffic_fraction": 0.0,
    "publishes_checkpoint": False,
    "replaces_default_policy": False,
    "connects_real_executor": False,
    "runs_new_ppo_update": False,
    "modifies_network": False,
    "modifies_action_space": False,
    "modifies_default_astar": False,
    "real_world_release_approved": False,
    "real_world_performance_claimed": False,
    "default_policy_replacement_approved": False,
    "real_executor_connection_approved": False,
    "default_policy_candidate_installation_approved": False,
}

FORBIDDEN_FIELDS = tuple(k for k, v in BOUNDARY_FIELDS.items() if v is False) + (
    "checkpoint_publication_approved",
    "final_release_approved",
    "performance_claimed",
    "relaxes_guard",
    "guard_relaxed",
    "modifies_network_or_action_space",
    "ackermann_feasible_trajectory_claimed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Global 99 controlled default-policy candidate installation preflight.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_global_99_controlled_default_policy_candidate_installation_preflight(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": summary["status"],
        "reason_codes": summary["reason_codes"],
        "controlled_installation_verdict": summary["controlled_installation_verdict"],
        "next_required_change": summary["next_required_change"],
        "summary": summary["summary"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_global_99_controlled_default_policy_candidate_installation_preflight(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_config(Path(config_path), repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)
    source_path = resolve_path(Path(config["source_sandbox_consumer_root"]), repo_root) / CONSUMER_SUMMARY_FILE
    read_reasons: list[str] = []
    source = _read_json(source_path, read_reasons, "sandbox_consumer")
    lineage = _lineage_audit(config, source, read_reasons)
    scope = _scope_audit(config)
    boundary = _boundary_audit(config, source)
    kill_switch = _simple_required_audit("kill-switch", config["require_kill_switch"] and source.get("candidate_load_audit_passed") is True)
    rollback = _simple_required_audit("rollback", config["require_rollback"] and source.get("rollback_audit_passed") is True)
    telemetry = _simple_required_audit("telemetry", config["require_telemetry"] and source.get("telemetry_audit_passed") is True)
    decision = _decision(lineage, scope, boundary, kill_switch, rollback, telemetry)
    generated_at = utc_now()
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_sandbox_consumer_status": source.get("status"),
        "source_sandbox_consumer_next_required_change": source.get("next_required_change"),
        "source_sandbox_consumer_verdict": source.get("sandbox_consumer_verdict"),
        "source_consumer_step_count": source.get("consumer_step_count"),
        "source_fallback_rate": source.get("fallback_rate"),
        "source_controlled_regression_count": source.get("controlled_regression_count"),
        "lineage_audit_passed": lineage["passed"],
        "review_scope_audit_passed": scope["passed"],
        "default_policy_read_only_audit_passed": scope["default_policy_read_only_audit_passed"],
        "executor_isolation_audit_passed": scope["executor_isolation_audit_passed"],
        "online_canary_audit_passed": scope["online_canary_audit_passed"],
        "kill_switch_audit_passed": kill_switch["passed"],
        "rollback_audit_passed": rollback["passed"],
        "telemetry_audit_passed": telemetry["passed"],
        "boundary_audit_passed": boundary["passed"],
        "controlled_default_policy_candidate_installation_preflight_passed": decision["status"] == "passed",
        "controlled_installation_verdict": decision["verdict"],
        "next_required_change": decision["next_required_change"],
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "report": str(paths["report"]),
        "source_sandbox_consumer_summary": str(source_path),
        "config": str(config_path),
        "output_root": str(output_root),
        "git_provenance": git_snapshot(repo_root),
        **BOUNDARY_FIELDS,
    }
    manifest = {
        "schema_version": "global-99-controlled-default-policy-candidate-installation-manifest/v1",
        "generated_at": generated_at,
        "source_sandbox_consumer_summary": str(source_path),
        "status": summary["status"],
        "next_required_change": summary["next_required_change"],
        **BOUNDARY_FIELDS,
    }
    rejection = {
        "schema_version": "global-99-controlled-installation-rejection-report/v1",
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
    }
    for path, payload in (
        (paths["manifest"], manifest),
        (paths["lineage"], lineage),
        (paths["scope"], scope),
        (paths["boundary"], boundary),
        (paths["kill_switch"], kill_switch),
        (paths["rollback"], rollback),
        (paths["telemetry"], telemetry),
        (paths["rejection_report"], rejection),
        (paths["summary"], summary),
    ):
        write_json(path, payload)
    paths["report"].write_text(_render_report(summary, rejection), encoding="utf-8")
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    normalized = dict(payload)
    if not isinstance(payload.get("source_sandbox_consumer_root"), str) or not payload["source_sandbox_consumer_root"].strip():
        raise ConfigError("source_sandbox_consumer_root must be a non-empty string")
    normalized["source_sandbox_consumer_root"] = str(resolve_path(Path(payload["source_sandbox_consumer_root"]), repo_root))
    for key in (
        "require_sandbox_consumer_passed",
        "require_installation_review_only",
        "require_default_policy_read_only",
        "require_executor_isolated",
        "require_online_canary_disabled",
        "require_kill_switch",
        "require_rollback",
        "require_telemetry",
        "controlled_installation_preflight_only",
    ):
        if not isinstance(payload.get(key), bool):
            raise ConfigError(f"{key} must be a boolean")
    normalized["canary_traffic_fraction"] = _float_value(payload.get("canary_traffic_fraction"), "canary_traffic_fraction")
    return normalized


def _paths(root: Path) -> dict[str, Path]:
    return {
        "summary": root / SUMMARY_FILE,
        "manifest": root / MANIFEST_FILE,
        "lineage": root / LINEAGE_AUDIT_FILE,
        "scope": root / SCOPE_AUDIT_FILE,
        "boundary": root / BOUNDARY_AUDIT_FILE,
        "kill_switch": root / KILL_SWITCH_AUDIT_FILE,
        "rollback": root / ROLLBACK_AUDIT_FILE,
        "telemetry": root / TELEMETRY_AUDIT_FILE,
        "rejection_report": root / REJECTION_REPORT_FILE,
        "report": root / REPORT_FILE,
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
    return payload if isinstance(payload, dict) else {}


def _lineage_audit(config: dict[str, Any], source: dict[str, Any], read_reasons: list[str]) -> dict[str, Any]:
    reasons = list(read_reasons)
    if source and config["require_sandbox_consumer_passed"]:
        if source.get("status") != "passed" or source.get("reason_codes") not in ([], None):
            reasons.append("sandbox_consumer_not_passed")
        if source.get("next_required_change") != "controlled_default_policy_candidate_installation_preflight":
            reasons.append("sandbox_consumer_wrong_next_required_change")
        if source.get("sandbox_consumer_verdict") != "eligible_for_controlled_default_policy_candidate_installation_preflight":
            reasons.append("sandbox_consumer_verdict_invalid")
    return {"schema_version": "global-99-controlled-installation-lineage-audit/v1", "passed": not reasons, "reason_codes": unique_sorted(reasons)}


def _scope_audit(config: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if config["require_installation_review_only"] is not True or config["controlled_installation_preflight_only"] is not True:
        reasons.append("controlled_installation_review_scope_invalid")
    default_policy = config["require_default_policy_read_only"] is True
    executor = config["require_executor_isolated"] is True
    online_canary = config["require_online_canary_disabled"] is True and config["canary_traffic_fraction"] == 0.0
    if not default_policy:
        reasons.append("default_policy_read_only_boundary_invalid")
    if not executor:
        reasons.append("executor_isolation_boundary_invalid")
    if not online_canary:
        reasons.append("online_canary_boundary_invalid")
    return {
        "schema_version": "global-99-controlled-installation-review-scope-audit/v1",
        "passed": not reasons,
        "reason_codes": reasons,
        "default_policy_read_only_audit_passed": default_policy,
        "executor_isolation_audit_passed": executor,
        "online_canary_audit_passed": online_canary,
    }


def _boundary_audit(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for field in FORBIDDEN_FIELDS:
        if source.get(field) is True:
            violations.append({"source": "sandbox_consumer", "field": field, "value": True})
    if _float_default(source.get("canary_traffic_fraction")) > 0:
        violations.append({"source": "sandbox_consumer", "field": "canary_traffic_fraction", "value": source.get("canary_traffic_fraction")})
    if config["canary_traffic_fraction"] > 0:
        violations.append({"source": "config", "field": "canary_traffic_fraction", "value": config["canary_traffic_fraction"]})
    return {
        "schema_version": "global-99-controlled-installation-boundary-audit/v1",
        "passed": not violations,
        "reason_codes": [] if not violations else ["controlled_installation_boundary_violation"],
        "violations": violations,
        **BOUNDARY_FIELDS,
    }


def _simple_required_audit(name: str, passed: bool) -> dict[str, Any]:
    return {
        "schema_version": f"global-99-controlled-installation-{name}-audit/v1",
        "passed": passed,
        "reason_codes": [] if passed else [f"{name}_audit_missing"],
    }


def _decision(*audits: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    for audit in audits:
        reasons.extend(audit.get("reason_codes", []))
    reasons = unique_sorted(reasons)
    status = "passed" if not reasons else "failed"
    if status == "passed":
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif any(reason.startswith("missing_sandbox_consumer") or reason.startswith("sandbox_consumer_") for reason in reasons):
        next_required_change = FIX_CONSUMER_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = REJECTION_NEXT_REQUIRED_CHANGE
    return {
        "status": status,
        "reason_codes": reasons,
        "verdict": PASS_VERDICT if status == "passed" else "blocked",
        "next_required_change": next_required_change,
    }


def _render_report(summary: dict[str, Any], rejection: dict[str, Any]) -> str:
    return "\n".join([
        "# Global 99 Controlled Default Policy Candidate Installation Preflight v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- verdict: `{summary['controlled_installation_verdict']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        "",
        "## Rejections",
        "",
        f"```json\n{json.dumps(rejection, ensure_ascii=False, indent=2)}\n```",
        "",
    ])


def _float_value(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{name} must be numeric")
    return float(value)


def _float_default(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
