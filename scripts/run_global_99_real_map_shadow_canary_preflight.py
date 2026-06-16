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


CONFIG_SCHEMA_VERSION = "global-99-real-map-shadow-canary-preflight-config/v1"
SUMMARY_SCHEMA_VERSION = "global-99-real-map-shadow-canary-preflight-summary/v1"
DEFAULT_CONFIG = "configs/global_99_real_map_shadow_canary_preflight_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_global_99_real_map_shadow_canary_preflight_v1"

RELEASE_SUMMARY_FILE = "global-99-real-map-release-governance-preflight-summary.json"
SHADOW_SUMMARY_FILE = "global-99-real-map-shadow-replay-summary.json"

SUMMARY_FILE = "global-99-real-map-shadow-canary-preflight-summary.json"
MANIFEST_FILE = "global-99-real-map-shadow-canary-manifest.json"
SHADOW_REPLAY_AUDIT_FILE = "global-99-real-map-shadow-replay-eligibility-audit.json"
CANARY_ELIGIBILITY_AUDIT_FILE = "global-99-real-map-canary-eligibility-audit.json"
BOUNDARY_AUDIT_FILE = "global-99-real-map-shadow-canary-boundary-audit.json"
KILL_SWITCH_AUDIT_FILE = "global-99-real-map-shadow-canary-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "global-99-real-map-shadow-canary-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "global-99-real-map-shadow-canary-telemetry-audit.json"
REJECTION_REPORT_FILE = "global-99-real-map-shadow-canary-rejection-report.json"
REPORT_FILE = "global-99-real-map-shadow-canary-preflight-report.md"

PASS_VERDICT = "eligible_for_global_99_real_map_shadow_canary_replay"
PASS_NEXT_REQUIRED_CHANGE = "global_99_real_map_shadow_canary_replay"
FIX_RELEASE_NEXT_REQUIRED_CHANGE = "fix_global_99_real_map_release_governance_preflight"
FIX_DETERMINISM_NEXT_REQUIRED_CHANGE = "fix_global_99_real_map_shadow_replay_determinism"
FIX_FALLBACK_NEXT_REQUIRED_CHANGE = "fix_global_99_real_map_shadow_canary_guard_fallback"
BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE = "resolve_global_99_real_map_shadow_canary_boundary_rejections"

BOUNDARY_FIELDS = {
    "shadow_mode": True,
    "canary_mode": True,
    "canary_traffic_fraction": 0.0,
    "starts_online_canary": False,
    "publishes_checkpoint": False,
    "replaces_default_policy": False,
    "connects_real_executor": False,
    "runs_new_ppo_update": False,
    "modifies_network": False,
    "modifies_action_space": False,
    "modifies_default_astar": False,
    "uses_path_planner": False,
    "real_world_release_approved": False,
    "real_world_performance_claimed": False,
    "default_policy_replacement_approved": False,
    "real_executor_connection_approved": False,
}

FORBIDDEN_FIELDS = tuple(k for k, v in BOUNDARY_FIELDS.items() if v is False and k != "uses_path_planner") + (
    "checkpoint_publication_approved",
    "final_release_approved",
    "performance_claimed",
    "relaxes_guard",
    "guard_relaxed",
    "modifies_network_or_action_space",
    "ackermann_feasible_trajectory_claimed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Global 99 Real Map Shadow Canary Preflight v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_global_99_real_map_shadow_canary_preflight(
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
        "real_map_shadow_canary_preflight_verdict": summary["real_map_shadow_canary_preflight_verdict"],
        "next_required_change": summary["next_required_change"],
        "summary": summary["summary"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_global_99_real_map_shadow_canary_preflight(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_config(Path(config_path), repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    source_paths = _source_paths(config, repo_root)
    evidence = _load_evidence(source_paths)
    shadow_replay = _shadow_replay_audit(config, evidence)
    canary = _canary_eligibility_audit(config, evidence)
    boundary = _boundary_audit(config, evidence)
    kill_switch = _simple_audit("kill-switch", shadow_replay["passed"] and boundary["passed"])
    rollback = _simple_audit("rollback", shadow_replay["passed"] and boundary["passed"])
    telemetry = _simple_audit("telemetry", shadow_replay["passed"] and canary["passed"])
    decision = _decision(evidence, shadow_replay, canary, boundary)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=Path(config_path),
        output_root=output_root,
        paths=paths,
        source_paths=source_paths,
        evidence=evidence,
        shadow_replay=shadow_replay,
        canary=canary,
        boundary=boundary,
        kill_switch=kill_switch,
        rollback=rollback,
        telemetry=telemetry,
        decision=decision,
        repo_root=Path(repo_root),
    )
    manifest = {
        "schema_version": "global-99-real-map-shadow-canary-manifest/v1",
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "artifacts": {k: str(v) for k, v in paths.items()},
        "status": summary["status"],
        "next_required_change": summary["next_required_change"],
    }
    rejection = {
        "schema_version": "global-99-real-map-shadow-canary-rejection-report/v1",
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
    }
    for key, payload in (
        ("shadow_replay", shadow_replay),
        ("canary", canary),
        ("boundary", boundary),
        ("kill_switch", kill_switch),
        ("rollback", rollback),
        ("telemetry", telemetry),
        ("rejection_report", rejection),
        ("manifest", manifest),
        ("summary", summary),
    ):
        write_json(paths[key], payload)
    paths["report"].write_text(_render_report(summary, rejection), encoding="utf-8")
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    normalized = dict(payload)
    for key in ("source_real_map_release_governance_root", "source_real_map_shadow_replay_root"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(payload[key]), repo_root))
    normalized["max_policy_guard_fallback_rate"] = _float_value(payload.get("max_policy_guard_fallback_rate"), "max_policy_guard_fallback_rate")
    normalized["canary_traffic_fraction"] = _float_value(payload.get("canary_traffic_fraction"), "canary_traffic_fraction")
    for key in ("shadow_mode", "canary_mode", "require_default_policy_authoritative", "require_release_governance_passed", "require_source_match_audit_passed", "require_no_open_grid_fallback", "require_boundary_closed"):
        if not isinstance(payload.get(key), bool):
            raise ConfigError(f"{key} must be a boolean")
        normalized[key] = payload[key]
    return normalized


def _source_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    release_root = resolve_path(Path(config["source_real_map_release_governance_root"]), repo_root)
    shadow_root = resolve_path(Path(config["source_real_map_shadow_replay_root"]), repo_root)
    return {
        "real_map_release_governance_root": release_root,
        "real_map_shadow_replay_root": shadow_root,
        "real_map_release_governance_summary": release_root / RELEASE_SUMMARY_FILE,
        "real_map_shadow_replay_summary": shadow_root / SHADOW_SUMMARY_FILE,
    }


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "shadow_replay": output_root / SHADOW_REPLAY_AUDIT_FILE,
        "canary": output_root / CANARY_ELIGIBILITY_AUDIT_FILE,
        "boundary": output_root / BOUNDARY_AUDIT_FILE,
        "kill_switch": output_root / KILL_SWITCH_AUDIT_FILE,
        "rollback": output_root / ROLLBACK_AUDIT_FILE,
        "telemetry": output_root / TELEMETRY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _load_evidence(source_paths: dict[str, Path]) -> dict[str, Any]:
    reasons: list[str] = []
    return {
        "release": _read_json(source_paths["real_map_release_governance_summary"], reasons, "real_map_release_governance"),
        "shadow": _read_json(source_paths["real_map_shadow_replay_summary"], reasons, "real_map_shadow_replay"),
        "read_reason_codes": unique_sorted(reasons),
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


def _shadow_replay_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    release = evidence["release"]
    shadow = evidence["shadow"]
    reasons = list(evidence["read_reason_codes"])
    if release and config["require_release_governance_passed"] and release.get("status") != "passed":
        reasons.append("real_map_release_governance_not_passed")
    if release and release.get("next_required_change") != "global_99_real_map_shadow_canary_preflight":
        reasons.append("real_map_release_governance_wrong_next_required_change")
    if config["require_source_match_audit_passed"] and release and not release.get("source_match_audit_passed", False):
        reasons.append("real_map_shadow_replay_source_match_failed")
    if _int_value(release.get("scenario_mismatch_count")) > 0 or _int_value(shadow.get("scenario_mismatch_count")) > 0:
        reasons.append("real_map_shadow_replay_source_match_failed")
    if config["require_no_open_grid_fallback"] and (release.get("open_grid_fallback_used") or shadow.get("open_grid_fallback_used")):
        reasons.append("real_map_open_grid_fallback_detected")
    return {
        "schema_version": "global-99-real-map-shadow-replay-eligibility-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "source_match_audit_passed": bool(release.get("source_match_audit_passed", False)),
        "scenario_mismatch_count": max(_int_value(release.get("scenario_mismatch_count")), _int_value(shadow.get("scenario_mismatch_count"))),
        "open_grid_fallback_used": bool(release.get("open_grid_fallback_used") or shadow.get("open_grid_fallback_used")),
    }


def _canary_eligibility_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    release = evidence["release"]
    reasons: list[str] = []
    fallback_rate = _float_default(release.get("policy_guard_fallback_rate"))
    if fallback_rate > config["max_policy_guard_fallback_rate"]:
        reasons.append("policy_guard_fallback_rate_too_high")
    if config["canary_traffic_fraction"] != 0.0:
        reasons.append("canary_traffic_fraction_must_be_zero")
    if not config["shadow_mode"] or not config["canary_mode"]:
        reasons.append("shadow_canary_mode_not_enabled")
    return {
        "schema_version": "global-99-real-map-canary-eligibility-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "policy_guard_fallback_rate": fallback_rate,
        "canary_traffic_fraction": config["canary_traffic_fraction"],
    }


def _boundary_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for source_name in ("release", "shadow"):
        source = evidence[source_name]
        for field in FORBIDDEN_FIELDS:
            if source.get(field) is True:
                violations.append({"source": source_name, "field": field, "value": True})
        if _float_default(source.get("canary_traffic_fraction")) > 0:
            violations.append({"source": source_name, "field": "canary_traffic_fraction", "value": source.get("canary_traffic_fraction")})
    if config["canary_traffic_fraction"] > 0:
        violations.append({"source": "config", "field": "canary_traffic_fraction", "value": config["canary_traffic_fraction"]})
    reasons = ["real_map_shadow_canary_boundary_violation"] if violations and config["require_boundary_closed"] else []
    return {
        "schema_version": "global-99-real-map-shadow-canary-boundary-audit/v1",
        "passed": not reasons,
        "reason_codes": reasons,
        "violations": violations,
        **BOUNDARY_FIELDS,
    }


def _simple_audit(name: str, passed: bool) -> dict[str, Any]:
    return {
        "schema_version": f"global-99-real-map-shadow-canary-{name}-audit/v1",
        "passed": bool(passed),
        "reason_codes": [] if passed else [f"{name}_audit_blocked"],
    }


def _decision(evidence: dict[str, Any], shadow: dict[str, Any], canary: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    reasons = unique_sorted(list(shadow["reason_codes"]) + list(canary["reason_codes"]) + list(boundary["reason_codes"]))
    status = "passed" if not reasons else "failed"
    next_required_change = PASS_NEXT_REQUIRED_CHANGE
    verdict = PASS_VERDICT
    if status == "failed":
        verdict = "blocked"
        if any(reason.startswith("missing_real_map_release_governance") or reason.startswith("real_map_release_governance") for reason in reasons):
            next_required_change = FIX_RELEASE_NEXT_REQUIRED_CHANGE
        elif "real_map_shadow_replay_source_match_failed" in reasons:
            next_required_change = FIX_DETERMINISM_NEXT_REQUIRED_CHANGE
        elif "policy_guard_fallback_rate_too_high" in reasons:
            next_required_change = FIX_FALLBACK_NEXT_REQUIRED_CHANGE
        elif "real_map_shadow_canary_boundary_violation" in reasons or "canary_traffic_fraction_must_be_zero" in reasons:
            next_required_change = BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE
        else:
            next_required_change = FIX_RELEASE_NEXT_REQUIRED_CHANGE
    return {"status": status, "reason_codes": reasons, "verdict": verdict, "next_required_change": next_required_change}


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    source_paths: dict[str, Path],
    evidence: dict[str, Any],
    shadow_replay: dict[str, Any],
    canary: dict[str, Any],
    boundary: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    release = evidence["release"]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_real_map_release_governance_status": release.get("status"),
        "source_real_map_release_governance_next_required_change": release.get("next_required_change"),
        "source_match_audit_passed": shadow_replay["source_match_audit_passed"],
        "scenario_mismatch_count": shadow_replay["scenario_mismatch_count"],
        "open_grid_fallback_used": shadow_replay["open_grid_fallback_used"],
        "policy_guard_fallback_rate": canary["policy_guard_fallback_rate"],
        "real_map_shadow_canary_preflight_passed": decision["status"] == "passed",
        "real_map_shadow_canary_preflight_verdict": decision["verdict"],
        "shadow_replay_eligibility_audit_passed": shadow_replay["passed"],
        "canary_eligibility_audit_passed": canary["passed"],
        "boundary_audit_passed": boundary["passed"],
        "kill_switch_audit_passed": kill_switch["passed"],
        "rollback_audit_passed": rollback["passed"],
        "telemetry_audit_passed": telemetry["passed"],
        "next_required_change": decision["next_required_change"],
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "report": str(paths["report"]),
        "source_paths": {key: str(value) for key, value in source_paths.items()},
        "config": str(config_path),
        "output_root": str(output_root),
        "git_provenance": git_snapshot(repo_root),
        **BOUNDARY_FIELDS,
    }
    return summary


def _render_report(summary: dict[str, Any], rejection: dict[str, Any]) -> str:
    return "\n".join([
        "# Global 99 Real Map Shadow Canary Preflight v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- verdict: `{summary['real_map_shadow_canary_preflight_verdict']}`",
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
