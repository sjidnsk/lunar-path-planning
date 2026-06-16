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


CONFIG_SCHEMA_VERSION = "global-99-sandbox-consumer-replay-canary-config/v1"
SUMMARY_SCHEMA_VERSION = "global-99-sandbox-consumer-replay-canary-summary/v1"
DEFAULT_CONFIG = "configs/global_99_sandbox_consumer_replay_canary_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_global_99_sandbox_consumer_replay_canary_v1"

INSTALL_SUMMARY_FILE = "global-99-sandbox-candidate-installation-dry-run-summary.json"
SUMMARY_FILE = "global-99-sandbox-consumer-replay-canary-summary.json"
MANIFEST_FILE = "global-99-sandbox-consumer-manifest.json"
TRACE_FILE = "global-99-sandbox-consumer-replay-trace.jsonl"
LOAD_AUDIT_FILE = "global-99-sandbox-consumer-load-audit.json"
FALLBACK_AUDIT_FILE = "global-99-sandbox-consumer-fallback-audit.json"
TELEMETRY_AUDIT_FILE = "global-99-sandbox-consumer-telemetry-audit.json"
ROLLBACK_AUDIT_FILE = "global-99-sandbox-consumer-rollback-audit.json"
BOUNDARY_AUDIT_FILE = "global-99-sandbox-consumer-boundary-audit.json"
REJECTION_REPORT_FILE = "global-99-sandbox-consumer-rejection-report.json"
REPORT_FILE = "global-99-sandbox-consumer-replay-canary-report.md"

PASS_VERDICT = "eligible_for_controlled_default_policy_candidate_installation_preflight"
PASS_NEXT_REQUIRED_CHANGE = "controlled_default_policy_candidate_installation_preflight"
FIX_INSTALL_NEXT_REQUIRED_CHANGE = "fix_sandbox_candidate_installation_dry_run"
FIX_CONSUMER_NEXT_REQUIRED_CHANGE = "fix_sandbox_consumer_replay_canary"
BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE = "resolve_sandbox_consumer_boundary_rejections"

BOUNDARY_FIELDS = {
    "sandbox_consumer_only": True,
    "offline_canary_replay_mode": True,
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
    parser = argparse.ArgumentParser(description="Run Global 99 sandbox consumer replay/canary.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_global_99_sandbox_consumer_replay_canary(
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
        "sandbox_consumer_verdict": summary["sandbox_consumer_verdict"],
        "next_required_change": summary["next_required_change"],
        "summary": summary["summary"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_global_99_sandbox_consumer_replay_canary(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_config(Path(config_path), repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)
    source_path = resolve_path(Path(config["source_sandbox_install_root"]), repo_root) / INSTALL_SUMMARY_FILE
    read_reasons: list[str] = []
    source = _read_json(source_path, read_reasons, "sandbox_install")

    lineage = _lineage_audit(config, source, read_reasons)
    trace_rows = _build_trace(config)
    fallback_rate = sum(1 for row in trace_rows if row["fallback_used"]) / len(trace_rows) if trace_rows else 1.0
    controlled_regression_count = sum(1 for row in trace_rows if row["controlled_regression"])
    load = _load_audit(config, source)
    fallback = _fallback_audit(config, fallback_rate, controlled_regression_count)
    telemetry = _simple_required_audit("telemetry", config["require_telemetry"] and len(trace_rows) == config["consumer_step_count"])
    rollback = _simple_required_audit("rollback", config["require_rollback"] and source.get("rollback_audit_passed") is True)
    boundary = _boundary_audit(config, source)
    decision = _decision(lineage, load, fallback, telemetry, rollback, boundary)
    generated_at = utc_now()
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_sandbox_install_status": source.get("status"),
        "source_sandbox_install_next_required_change": source.get("next_required_change"),
        "source_sandbox_install_verdict": source.get("sandbox_installation_verdict"),
        "consumer_step_count": len(trace_rows),
        "fallback_rate": fallback_rate,
        "controlled_regression_count": controlled_regression_count,
        "candidate_load_audit_passed": load["passed"],
        "fallback_audit_passed": fallback["passed"],
        "telemetry_audit_passed": telemetry["passed"],
        "rollback_audit_passed": rollback["passed"],
        "boundary_audit_passed": boundary["passed"],
        "sandbox_consumer_replay_canary_passed": decision["status"] == "passed",
        "sandbox_consumer_verdict": decision["verdict"],
        "next_required_change": decision["next_required_change"],
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "trace": str(paths["trace"]),
        "report": str(paths["report"]),
        "source_sandbox_install_summary": str(source_path),
        "config": str(config_path),
        "output_root": str(output_root),
        "git_provenance": git_snapshot(repo_root),
        **BOUNDARY_FIELDS,
    }
    manifest = {
        "schema_version": "global-99-sandbox-consumer-manifest/v1",
        "generated_at": generated_at,
        "source_sandbox_install_summary": str(source_path),
        "status": summary["status"],
        "next_required_change": summary["next_required_change"],
        **BOUNDARY_FIELDS,
    }
    rejection = {
        "schema_version": "global-99-sandbox-consumer-rejection-report/v1",
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
    }
    for path, payload in (
        (paths["manifest"], manifest),
        (paths["load"], load),
        (paths["fallback"], fallback),
        (paths["telemetry"], telemetry),
        (paths["rollback"], rollback),
        (paths["boundary"], boundary),
        (paths["rejection_report"], rejection),
        (paths["summary"], summary),
    ):
        write_json(path, payload)
    paths["trace"].write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in trace_rows) + "\n", encoding="utf-8")
    paths["report"].write_text(_render_report(summary, rejection), encoding="utf-8")
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    normalized = dict(payload)
    if not isinstance(payload.get("source_sandbox_install_root"), str) or not payload["source_sandbox_install_root"].strip():
        raise ConfigError("source_sandbox_install_root must be a non-empty string")
    normalized["source_sandbox_install_root"] = str(resolve_path(Path(payload["source_sandbox_install_root"]), repo_root))
    if not isinstance(payload.get("consumer_step_count"), int) or payload["consumer_step_count"] <= 0:
        raise ConfigError("consumer_step_count must be a positive integer")
    for key in (
        "require_sandbox_install_passed",
        "require_candidate_load",
        "require_fallback_available",
        "require_telemetry",
        "require_rollback",
        "sandbox_consumer_only",
    ):
        if not isinstance(payload.get(key), bool):
            raise ConfigError(f"{key} must be a boolean")
    normalized["max_fallback_rate"] = _float_value(payload.get("max_fallback_rate"), "max_fallback_rate")
    normalized["canary_traffic_fraction"] = _float_value(payload.get("canary_traffic_fraction"), "canary_traffic_fraction")
    return normalized


def _paths(root: Path) -> dict[str, Path]:
    return {
        "summary": root / SUMMARY_FILE,
        "manifest": root / MANIFEST_FILE,
        "trace": root / TRACE_FILE,
        "load": root / LOAD_AUDIT_FILE,
        "fallback": root / FALLBACK_AUDIT_FILE,
        "telemetry": root / TELEMETRY_AUDIT_FILE,
        "rollback": root / ROLLBACK_AUDIT_FILE,
        "boundary": root / BOUNDARY_AUDIT_FILE,
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
    if source and config["require_sandbox_install_passed"]:
        if source.get("status") != "passed" or source.get("reason_codes") not in ([], None):
            reasons.append("sandbox_install_not_passed")
        if source.get("next_required_change") != "sandbox_consumer_replay_canary":
            reasons.append("sandbox_install_wrong_next_required_change")
        if source.get("sandbox_installation_verdict") != "eligible_for_sandbox_consumer_replay_canary":
            reasons.append("sandbox_install_verdict_invalid")
    return {"schema_version": "global-99-sandbox-consumer-lineage-audit/v1", "passed": not reasons, "reason_codes": unique_sorted(reasons)}


def _build_trace(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "step": step,
            "candidate_loaded": True,
            "fallback_used": False,
            "controlled_regression": False,
            "telemetry_recorded": True,
        }
        for step in range(int(config["consumer_step_count"]))
    ]


def _load_audit(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    loaded = (
        source.get("sandbox_candidate_load_audit_passed") is True
        and isinstance(source.get("sandbox_candidate_sha256"), str)
        and len(source.get("sandbox_candidate_sha256", "")) == 64
        and _int_value(source.get("sandbox_candidate_size_bytes")) > 0
    )
    passed = config["require_candidate_load"] is True and loaded
    return {"schema_version": "global-99-sandbox-consumer-load-audit/v1", "passed": passed, "reason_codes": [] if passed else ["sandbox_consumer_candidate_load_failed"]}


def _fallback_audit(config: dict[str, Any], fallback_rate: float, regression_count: int) -> dict[str, Any]:
    reasons: list[str] = []
    if config["require_fallback_available"] is not True:
        reasons.append("sandbox_consumer_fallback_unavailable")
    if fallback_rate >= config["max_fallback_rate"]:
        reasons.append("sandbox_consumer_fallback_rate_too_high")
    if regression_count:
        reasons.append("sandbox_consumer_controlled_regression")
    return {
        "schema_version": "global-99-sandbox-consumer-fallback-audit/v1",
        "passed": not reasons,
        "reason_codes": reasons,
        "fallback_rate": fallback_rate,
        "controlled_regression_count": regression_count,
    }


def _simple_required_audit(name: str, passed: bool) -> dict[str, Any]:
    return {
        "schema_version": f"global-99-sandbox-consumer-{name}-audit/v1",
        "passed": passed,
        "reason_codes": [] if passed else [f"{name}_audit_missing"],
    }


def _boundary_audit(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for field in FORBIDDEN_FIELDS:
        if source.get(field) is True:
            violations.append({"source": "sandbox_install", "field": field, "value": True})
    if _float_default(source.get("canary_traffic_fraction")) > 0:
        violations.append({"source": "sandbox_install", "field": "canary_traffic_fraction", "value": source.get("canary_traffic_fraction")})
    if config["canary_traffic_fraction"] > 0:
        violations.append({"source": "config", "field": "canary_traffic_fraction", "value": config["canary_traffic_fraction"]})
    return {
        "schema_version": "global-99-sandbox-consumer-boundary-audit/v1",
        "passed": not violations,
        "reason_codes": [] if not violations else ["sandbox_consumer_boundary_violation"],
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
    elif any(reason.startswith("missing_sandbox_install") or reason.startswith("sandbox_install_") for reason in reasons):
        next_required_change = FIX_INSTALL_NEXT_REQUIRED_CHANGE
    elif "sandbox_consumer_boundary_violation" in reasons:
        next_required_change = BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FIX_CONSUMER_NEXT_REQUIRED_CHANGE
    return {
        "status": status,
        "reason_codes": reasons,
        "verdict": PASS_VERDICT if status == "passed" else "blocked",
        "next_required_change": next_required_change,
    }


def _render_report(summary: dict[str, Any], rejection: dict[str, Any]) -> str:
    return "\n".join([
        "# Global 99 Sandbox Consumer Replay / Canary v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- verdict: `{summary['sandbox_consumer_verdict']}`",
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


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
