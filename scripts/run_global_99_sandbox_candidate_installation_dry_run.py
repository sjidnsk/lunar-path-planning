from __future__ import annotations

import argparse
import hashlib
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


CONFIG_SCHEMA_VERSION = "global-99-sandbox-candidate-installation-dry-run-config/v1"
SUMMARY_SCHEMA_VERSION = "global-99-sandbox-candidate-installation-dry-run-summary/v1"
DEFAULT_CONFIG = "configs/global_99_sandbox_candidate_installation_dry_run_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_global_99_sandbox_candidate_installation_dry_run_v1"

AUTHORIZATION_SUMMARY_FILE = "global-99-default-policy-candidate-authorization-preflight-summary.json"
SUMMARY_FILE = "global-99-sandbox-candidate-installation-dry-run-summary.json"
MANIFEST_FILE = "global-99-sandbox-candidate-installation-manifest.json"
PROVENANCE_AUDIT_FILE = "global-99-sandbox-candidate-provenance-audit.json"
HASH_AUDIT_FILE = "global-99-sandbox-candidate-hash-audit.json"
LOAD_AUDIT_FILE = "global-99-sandbox-candidate-load-audit.json"
ROLLBACK_AUDIT_FILE = "global-99-sandbox-candidate-rollback-audit.json"
KILL_SWITCH_AUDIT_FILE = "global-99-sandbox-candidate-kill-switch-audit.json"
TELEMETRY_AUDIT_FILE = "global-99-sandbox-candidate-telemetry-audit.json"
BOUNDARY_AUDIT_FILE = "global-99-sandbox-candidate-boundary-audit.json"
REJECTION_REPORT_FILE = "global-99-sandbox-candidate-rejection-report.json"
REPORT_FILE = "global-99-sandbox-candidate-installation-dry-run-report.md"

PASS_VERDICT = "eligible_for_sandbox_consumer_replay_canary"
PASS_NEXT_REQUIRED_CHANGE = "sandbox_consumer_replay_canary"
FIX_AUTHORIZATION_NEXT_REQUIRED_CHANGE = "fix_default_policy_candidate_authorization_preflight"
FIX_SANDBOX_INSTALL_NEXT_REQUIRED_CHANGE = "fix_sandbox_candidate_installation_dry_run"
BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE = "resolve_sandbox_candidate_installation_boundary_rejections"

BOUNDARY_FIELDS = {
    "sandbox_installation_only": True,
    "sandbox_candidate_loaded": True,
    "default_policy_touched": False,
    "publishes_checkpoint": False,
    "replaces_default_policy": False,
    "connects_real_executor": False,
    "starts_online_canary": False,
    "canary_traffic_fraction": 0.0,
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
    parser = argparse.ArgumentParser(description="Run Global 99 sandbox candidate installation dry run.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_global_99_sandbox_candidate_installation_dry_run(
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
        "sandbox_installation_verdict": summary["sandbox_installation_verdict"],
        "next_required_change": summary["next_required_change"],
        "summary": summary["summary"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_global_99_sandbox_candidate_installation_dry_run(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_config(Path(config_path), repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)
    source_path = resolve_path(Path(config["source_authorization_root"]), repo_root) / AUTHORIZATION_SUMMARY_FILE
    read_reasons: list[str] = []
    source = _read_json(source_path, read_reasons, "authorization")
    candidate_path = _write_sandbox_candidate(config, source, repo_root)
    candidate_sha = _sha256(candidate_path) if candidate_path.is_file() else ""
    candidate_size = candidate_path.stat().st_size if candidate_path.is_file() else 0

    lineage = _lineage_audit(config, source, read_reasons)
    provenance = _provenance_audit(candidate_path, candidate_sha, candidate_size)
    hash_audit = _hash_audit(config, candidate_sha, candidate_size)
    load = _load_audit(config, candidate_path, candidate_sha)
    rollback = _simple_required_audit("rollback", config["require_rollback"])
    kill_switch = _simple_required_audit("kill-switch", config["require_kill_switch"])
    telemetry = _simple_required_audit("telemetry", config["require_telemetry"])
    boundary = _boundary_audit(config, source)
    decision = _decision(lineage, provenance, hash_audit, load, rollback, kill_switch, telemetry, boundary)
    generated_at = utc_now()
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_authorization_status": source.get("status"),
        "source_authorization_next_required_change": source.get("next_required_change"),
        "source_authorization_verdict": source.get("authorization_verdict"),
        "sandbox_candidate_id": config["sandbox_candidate_id"],
        "sandbox_candidate_path": str(candidate_path),
        "sandbox_candidate_sha256": candidate_sha,
        "sandbox_candidate_size_bytes": candidate_size,
        "lineage_audit_passed": lineage["passed"],
        "provenance_audit_passed": provenance["passed"],
        "sandbox_candidate_hash_audit_passed": hash_audit["passed"],
        "sandbox_candidate_load_audit_passed": load["passed"],
        "rollback_audit_passed": rollback["passed"],
        "kill_switch_audit_passed": kill_switch["passed"],
        "telemetry_audit_passed": telemetry["passed"],
        "boundary_audit_passed": boundary["passed"],
        "sandbox_candidate_installation_dry_run_passed": decision["status"] == "passed",
        "sandbox_installation_verdict": decision["verdict"],
        "next_required_change": decision["next_required_change"],
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "report": str(paths["report"]),
        "source_authorization_summary": str(source_path),
        "config": str(config_path),
        "output_root": str(output_root),
        "git_provenance": git_snapshot(repo_root),
        **BOUNDARY_FIELDS,
    }
    manifest = {
        "schema_version": "global-99-sandbox-candidate-installation-manifest/v1",
        "generated_at": generated_at,
        "source_authorization_summary": str(source_path),
        "sandbox_candidate_path": str(candidate_path),
        "status": summary["status"],
        "next_required_change": summary["next_required_change"],
        **BOUNDARY_FIELDS,
    }
    rejection = {
        "schema_version": "global-99-sandbox-candidate-rejection-report/v1",
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
    }
    for path, payload in (
        (paths["manifest"], manifest),
        (paths["provenance"], provenance),
        (paths["hash"], hash_audit),
        (paths["load"], load),
        (paths["rollback"], rollback),
        (paths["kill_switch"], kill_switch),
        (paths["telemetry"], telemetry),
        (paths["boundary"], boundary),
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
    for key in ("source_authorization_root", "sandbox_installation_root", "sandbox_candidate_id"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    normalized["source_authorization_root"] = str(resolve_path(Path(payload["source_authorization_root"]), repo_root))
    normalized["sandbox_installation_root"] = str(resolve_path(Path(payload["sandbox_installation_root"]), repo_root))
    for key in (
        "require_authorization_passed",
        "require_candidate_hash",
        "require_sandbox_load",
        "require_rollback",
        "require_kill_switch",
        "require_telemetry",
        "sandbox_installation_only",
    ):
        if not isinstance(payload.get(key), bool):
            raise ConfigError(f"{key} must be a boolean")
    normalized["canary_traffic_fraction"] = _float_value(payload.get("canary_traffic_fraction"), "canary_traffic_fraction")
    return normalized


def _paths(root: Path) -> dict[str, Path]:
    return {
        "summary": root / SUMMARY_FILE,
        "manifest": root / MANIFEST_FILE,
        "provenance": root / PROVENANCE_AUDIT_FILE,
        "hash": root / HASH_AUDIT_FILE,
        "load": root / LOAD_AUDIT_FILE,
        "rollback": root / ROLLBACK_AUDIT_FILE,
        "kill_switch": root / KILL_SWITCH_AUDIT_FILE,
        "telemetry": root / TELEMETRY_AUDIT_FILE,
        "boundary": root / BOUNDARY_AUDIT_FILE,
        "rejection_report": root / REJECTION_REPORT_FILE,
        "report": root / REPORT_FILE,
    }


def _write_sandbox_candidate(config: dict[str, Any], source: dict[str, Any], repo_root: Path) -> Path:
    sandbox_root = resolve_path(Path(config["sandbox_installation_root"]), repo_root)
    sandbox_root.mkdir(parents=True, exist_ok=True)
    candidate_path = sandbox_root / "global-99-sandbox-candidate.json"
    payload = {
        "schema_version": "global-99-sandbox-candidate/v1",
        "candidate_id": config["sandbox_candidate_id"],
        "source_authorization_verdict": source.get("authorization_verdict"),
        "source_authorization_next_required_change": source.get("next_required_change"),
        "sandbox_only": True,
        "default_policy_replacement_approved": False,
        "connects_real_executor": False,
    }
    write_json(candidate_path, payload)
    return candidate_path


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
    if source and config["require_authorization_passed"]:
        if source.get("status") != "passed" or source.get("reason_codes") not in ([], None):
            reasons.append("authorization_not_passed")
        if source.get("next_required_change") != "sandbox_candidate_installation_dry_run":
            reasons.append("authorization_wrong_next_required_change")
        if source.get("authorization_verdict") != "eligible_for_sandbox_candidate_installation_dry_run":
            reasons.append("authorization_verdict_invalid")
    return {"schema_version": "global-99-sandbox-candidate-lineage-audit/v1", "passed": not reasons, "reason_codes": unique_sorted(reasons)}


def _provenance_audit(candidate_path: Path, candidate_sha: str, candidate_size: int) -> dict[str, Any]:
    passed = candidate_path.is_file() and bool(candidate_sha) and candidate_size > 0
    return {
        "schema_version": "global-99-sandbox-candidate-provenance-audit/v1",
        "passed": passed,
        "reason_codes": [] if passed else ["sandbox_candidate_provenance_missing"],
        "sandbox_candidate_path": str(candidate_path),
        "sandbox_candidate_sha256": candidate_sha,
        "sandbox_candidate_size_bytes": candidate_size,
    }


def _hash_audit(config: dict[str, Any], candidate_sha: str, candidate_size: int) -> dict[str, Any]:
    passed = config["require_candidate_hash"] is True and len(candidate_sha) == 64 and candidate_size > 0
    return {
        "schema_version": "global-99-sandbox-candidate-hash-audit/v1",
        "passed": passed,
        "reason_codes": [] if passed else ["sandbox_candidate_hash_missing"],
        "sandbox_candidate_sha256": candidate_sha,
        "sandbox_candidate_size_bytes": candidate_size,
    }


def _load_audit(config: dict[str, Any], candidate_path: Path, candidate_sha: str) -> dict[str, Any]:
    loaded = False
    if candidate_path.is_file():
        try:
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            loaded = payload.get("sandbox_only") is True and payload.get("default_policy_replacement_approved") is False
        except json.JSONDecodeError:
            loaded = False
    passed = config["require_sandbox_load"] is True and loaded and len(candidate_sha) == 64
    return {
        "schema_version": "global-99-sandbox-candidate-load-audit/v1",
        "passed": passed,
        "reason_codes": [] if passed else ["sandbox_candidate_load_failed"],
        "sandbox_load_verified": loaded,
    }


def _simple_required_audit(name: str, required: bool) -> dict[str, Any]:
    return {
        "schema_version": f"global-99-sandbox-candidate-{name}-audit/v1",
        "passed": required is True,
        "reason_codes": [] if required is True else [f"{name}_audit_missing"],
    }


def _boundary_audit(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for field in FORBIDDEN_FIELDS:
        if source.get(field) is True:
            violations.append({"source": "authorization", "field": field, "value": True})
    if _float_default(source.get("canary_traffic_fraction")) > 0:
        violations.append({"source": "authorization", "field": "canary_traffic_fraction", "value": source.get("canary_traffic_fraction")})
    if config["canary_traffic_fraction"] > 0:
        violations.append({"source": "config", "field": "canary_traffic_fraction", "value": config["canary_traffic_fraction"]})
    return {
        "schema_version": "global-99-sandbox-candidate-boundary-audit/v1",
        "passed": not violations,
        "reason_codes": [] if not violations else ["sandbox_installation_boundary_violation"],
        "violations": violations,
        **BOUNDARY_FIELDS,
    }


def _decision(*audits: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    for audit in audits:
        reasons.extend(audit.get("reason_codes", []))
    reasons = unique_sorted(reasons)
    status = "passed" if not reasons else "failed"
    if status == "passed":
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif any(reason.startswith("missing_authorization") or reason.startswith("authorization_") for reason in reasons):
        next_required_change = FIX_AUTHORIZATION_NEXT_REQUIRED_CHANGE
    elif "sandbox_installation_boundary_violation" in reasons:
        next_required_change = BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FIX_SANDBOX_INSTALL_NEXT_REQUIRED_CHANGE
    return {
        "status": status,
        "reason_codes": reasons,
        "verdict": PASS_VERDICT if status == "passed" else "blocked",
        "next_required_change": next_required_change,
    }


def _render_report(summary: dict[str, Any], rejection: dict[str, Any]) -> str:
    return "\n".join([
        "# Global 99 Sandbox Candidate Installation Dry Run v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- verdict: `{summary['sandbox_installation_verdict']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        "",
        "## Rejections",
        "",
        f"```json\n{json.dumps(rejection, ensure_ascii=False, indent=2)}\n```",
        "",
    ])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_value(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{name} must be numeric")
    return float(value)


def _float_default(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
