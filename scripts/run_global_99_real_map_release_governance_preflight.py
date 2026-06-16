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


CONFIG_SCHEMA_VERSION = "global-99-real-map-release-governance-preflight-config/v1"
SUMMARY_SCHEMA_VERSION = "global-99-real-map-release-governance-preflight-summary/v1"
MANIFEST_SCHEMA_VERSION = "global-99-real-map-release-governance-manifest/v1"

DEFAULT_CONFIG = "configs/global_99_real_map_release_governance_preflight_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_global_99_real_map_release_governance_preflight_v1"

SHADOW_SUMMARY_FILE = "global-99-real-map-shadow-replay-summary.json"
PREFLIGHT_SUMMARY_FILE = "global-99-real-map-preflight-summary.json"
DOMAIN_GAP_SUMMARY_FILE = "quasi-real-map-domain-gap-summary.json"
PATH_FEEDBACK_SUMMARY_FILE = "quasi-real-map-path-feedback-summary.json"

SUMMARY_FILE = "global-99-real-map-release-governance-preflight-summary.json"
MANIFEST_FILE = "global-99-real-map-release-governance-manifest.json"
LINEAGE_AUDIT_FILE = "global-99-real-map-release-evidence-lineage-audit.json"
SHADOW_REPLAY_AUDIT_FILE = "global-99-real-map-release-shadow-replay-audit.json"
PREFLIGHT_AUDIT_FILE = "global-99-real-map-release-real-map-preflight-audit.json"
DOMAIN_GAP_AUDIT_FILE = "global-99-real-map-release-domain-gap-audit.json"
PATH_FEEDBACK_AUDIT_FILE = "global-99-real-map-release-path-feedback-audit.json"
SCOPE_AUDIT_FILE = "global-99-real-map-release-scope-audit.json"
BOUNDARY_AUDIT_FILE = "global-99-real-map-release-boundary-audit.json"
KILL_SWITCH_AUDIT_FILE = "global-99-real-map-release-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "global-99-real-map-release-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "global-99-real-map-release-telemetry-audit.json"
REJECTION_REPORT_FILE = "global-99-real-map-release-rejection-report.json"
REPORT_FILE = "global-99-real-map-release-governance-preflight-report.md"

PASS_VERDICT = "eligible_for_global_99_real_map_shadow_canary_preflight"
PASS_NEXT_REQUIRED_CHANGE = "global_99_real_map_shadow_canary_preflight"
FIX_SHADOW_NEXT_REQUIRED_CHANGE = "fix_global_99_real_map_shadow_replay"
FIX_PREFLIGHT_NEXT_REQUIRED_CHANGE = "fix_global_99_real_map_preflight"
FIX_DOMAIN_GAP_NEXT_REQUIRED_CHANGE = "fix_quasi_real_map_domain_gap_evidence"
FIX_DETERMINISM_NEXT_REQUIRED_CHANGE = "fix_global_99_real_map_shadow_replay_determinism"
FIX_CONTEXT_NEXT_REQUIRED_CHANGE = "fix_real_map_context_identity"
FIX_PATH_FEEDBACK_NEXT_REQUIRED_CHANGE = "fix_real_map_path_feedback_contract"
FIX_FALLBACK_NEXT_REQUIRED_CHANGE = "fix_global_99_shadow_canary_guard_fallback"
BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE = "resolve_global_99_real_map_release_governance_boundary_rejections"

BOUNDARY_FIELDS = {
    "real_map_release_governance_preflight_only": True,
    "uses_lola_quasi_real_roi": True,
    "uses_path_feedback_sidecar": True,
    "audits_offline_path_feedback_replay": True,
    "audits_offline_path_planner_route_replay": True,
    "uses_path_planner": False,
    "path_planner_use_scope": "not_invoked_by_release_governance_preflight",
    "audited_path_planner_use_scope": "offline_path_feedback_replay_only",
    "real_world_release_approved": False,
    "real_world_performance_claimed": False,
    "publishes_checkpoint": False,
    "replaces_default_policy": False,
    "connects_real_executor": False,
    "starts_online_canary": False,
    "canary_traffic_fraction": 0.0,
    "runs_new_ppo_update": False,
    "modifies_network": False,
    "modifies_action_space": False,
    "modifies_default_astar": False,
    "default_policy_replacement_approved": False,
    "real_executor_connection_approved": False,
}

FORBIDDEN_BOOLEAN_FIELDS = (
    "real_world_release_approved",
    "real_world_performance_claimed",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "runs_new_ppo_update",
    "runs_ppo_update",
    "modifies_network",
    "modifies_action_space",
    "modifies_default_astar",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "checkpoint_publication_approved",
    "final_release_approved",
    "performance_claimed",
    "relaxes_guard",
    "guard_relaxed",
    "modifies_network_or_action_space",
    "ackermann_feasible_trajectory_claimed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Global 99 Real Map Release Governance Preflight v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_global_99_real_map_release_governance_preflight(
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
                "real_map_release_governance_verdict": summary["real_map_release_governance_verdict"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_global_99_real_map_release_governance_preflight(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(Path(config_path), repo_root)
    source_paths = _source_paths(config, repo_root)
    paths = _artifact_paths(output_root)
    evidence = _load_evidence(source_paths)

    lineage = _lineage_audit(source_paths, evidence)
    shadow = _shadow_replay_audit(config, evidence)
    preflight = _preflight_audit(config, evidence)
    domain_gap = _domain_gap_audit(config, evidence)
    path_feedback = _path_feedback_audit(config, evidence)
    scope = _scope_audit(config, evidence)
    boundary = _boundary_audit(config, evidence)
    kill_switch = _simple_audit("kill_switch", lineage["passed"] and boundary["passed"])
    rollback = _simple_audit("rollback", lineage["passed"] and boundary["passed"])
    telemetry = _simple_audit("telemetry", lineage["passed"] and shadow["passed"] and path_feedback["passed"])
    decision = _decision(config, evidence, lineage, shadow, preflight, domain_gap, path_feedback, scope, boundary)

    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=Path(config_path),
        output_root=output_root,
        paths=paths,
        source_paths=source_paths,
        evidence=evidence,
        lineage=lineage,
        shadow=shadow,
        preflight=preflight,
        domain_gap=domain_gap,
        path_feedback=path_feedback,
        scope=scope,
        boundary=boundary,
        kill_switch=kill_switch,
        rollback=rollback,
        telemetry=telemetry,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, Path(config_path), output_root, paths, summary)
    rejection = _rejection_report(generated_at, decision, lineage, shadow, preflight, domain_gap, path_feedback, boundary)

    for key, payload in (
        ("lineage", lineage),
        ("shadow_replay", shadow),
        ("preflight", preflight),
        ("domain_gap", domain_gap),
        ("path_feedback", path_feedback),
        ("scope", scope),
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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for key in ("source_real_map_shadow_replay_root", "source_real_map_preflight_root", "source_quasi_real_domain_gap_root"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    normalized = dict(payload)
    for key in ("source_real_map_shadow_replay_root", "source_real_map_preflight_root", "source_quasi_real_domain_gap_root"):
        normalized[key] = str(resolve_path(Path(payload[key]), repo_root))
    normalized["target_coverage_rate"] = _float_value(payload.get("target_coverage_rate"), "target_coverage_rate")
    normalized["max_policy_guard_fallback_rate"] = _float_value(
        payload.get("max_policy_guard_fallback_rate"),
        "max_policy_guard_fallback_rate",
    )
    normalized["source_match_coverage_tolerance"] = _float_value(
        payload.get("source_match_coverage_tolerance"),
        "source_match_coverage_tolerance",
    )
    normalized["source_match_path_cost_tolerance_m"] = _float_value(
        payload.get("source_match_path_cost_tolerance_m"),
        "source_match_path_cost_tolerance_m",
    )
    for key in (
        "require_real_map_shadow_replay_passed",
        "require_real_map_preflight_passed",
        "require_domain_gap_acceptable",
        "require_source_match_audit_passed",
        "require_context_ids",
        "require_no_open_grid_fallback",
        "require_release_boundaries_closed",
    ):
        normalized[key] = _bool_value(payload.get(key), key)
    return normalized


def _source_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    shadow_root = resolve_path(Path(config["source_real_map_shadow_replay_root"]), repo_root)
    preflight_root = resolve_path(Path(config["source_real_map_preflight_root"]), repo_root)
    domain_root = resolve_path(Path(config["source_quasi_real_domain_gap_root"]), repo_root)
    return {
        "real_map_shadow_replay_root": shadow_root,
        "real_map_preflight_root": preflight_root,
        "quasi_real_domain_gap_root": domain_root,
        "real_map_shadow_replay_summary": shadow_root / SHADOW_SUMMARY_FILE,
        "real_map_preflight_summary": preflight_root / PREFLIGHT_SUMMARY_FILE,
        "domain_gap_summary": domain_root / DOMAIN_GAP_SUMMARY_FILE,
        "path_feedback_summary": domain_root / PATH_FEEDBACK_SUMMARY_FILE,
    }


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "lineage": output_root / LINEAGE_AUDIT_FILE,
        "shadow_replay": output_root / SHADOW_REPLAY_AUDIT_FILE,
        "preflight": output_root / PREFLIGHT_AUDIT_FILE,
        "domain_gap": output_root / DOMAIN_GAP_AUDIT_FILE,
        "path_feedback": output_root / PATH_FEEDBACK_AUDIT_FILE,
        "scope": output_root / SCOPE_AUDIT_FILE,
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
        "shadow": _read_json(source_paths["real_map_shadow_replay_summary"], reasons, "real_map_shadow_replay"),
        "preflight": _read_json(source_paths["real_map_preflight_summary"], reasons, "real_map_preflight"),
        "domain_gap": _read_json(source_paths["domain_gap_summary"], reasons, "domain_gap"),
        "path_feedback": _read_json(source_paths["path_feedback_summary"], reasons, "quasi_real_path_feedback"),
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
    if not isinstance(payload, dict):
        reasons.append(f"invalid_{label}_summary")
        return {}
    return payload


def _lineage_audit(source_paths: dict[str, Path], evidence: dict[str, Any]) -> dict[str, Any]:
    reasons = list(evidence["read_reason_codes"])
    return {
        "schema_version": "global-99-real-map-release-evidence-lineage-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "source_paths": {key: str(value) for key, value in source_paths.items()},
    }


def _shadow_replay_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    shadow = evidence["shadow"]
    reasons: list[str] = []
    if not shadow:
        reasons.append("missing_real_map_shadow_replay_summary")
    if shadow and config["require_real_map_shadow_replay_passed"] and shadow.get("status") != "passed":
        reasons.append("real_map_shadow_replay_not_passed")
    if shadow and shadow.get("next_required_change") != "global_99_real_map_release_governance_preflight":
        reasons.append("real_map_shadow_replay_wrong_next_required_change")
    if config["require_source_match_audit_passed"] and shadow and not shadow.get("source_match_audit_passed", False):
        reasons.append("real_map_shadow_replay_source_match_failed")
    if _int_value(shadow.get("scenario_mismatch_count")) > 0:
        reasons.append("real_map_shadow_replay_source_match_failed")
    if _float_default(shadow.get("max_replay_coverage_delta")) > config["source_match_coverage_tolerance"]:
        reasons.append("real_map_shadow_replay_source_match_failed")
    if _float_default(shadow.get("max_replay_path_cost_delta_m")) > config["source_match_path_cost_tolerance_m"]:
        reasons.append("real_map_shadow_replay_source_match_failed")
    return {
        "schema_version": "global-99-real-map-release-shadow-replay-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "status": shadow.get("status"),
        "next_required_change": shadow.get("next_required_change"),
        "source_match_audit_passed": bool(shadow.get("source_match_audit_passed", False)),
        "scenario_mismatch_count": _int_value(shadow.get("scenario_mismatch_count")),
        "max_replay_coverage_delta": _float_default(shadow.get("max_replay_coverage_delta")),
        "max_replay_path_cost_delta_m": _float_default(shadow.get("max_replay_path_cost_delta_m")),
        "path_planner_use_scope": shadow.get("path_planner_use_scope"),
    }


def _preflight_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    preflight = evidence["preflight"]
    reasons: list[str] = []
    if not preflight:
        reasons.append("missing_real_map_preflight_summary")
    if preflight and config["require_real_map_preflight_passed"] and preflight.get("status") != "passed":
        reasons.append("real_map_preflight_not_passed")
    if preflight and preflight.get("next_required_change") != "global_99_real_map_shadow_replay":
        reasons.append("real_map_preflight_wrong_next_required_change")
    return {
        "schema_version": "global-99-real-map-release-real-map-preflight-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "status": preflight.get("status"),
        "next_required_change": preflight.get("next_required_change"),
        "real_map_preflight_verdict": preflight.get("real_map_preflight_verdict"),
    }


def _domain_gap_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    domain = evidence["domain_gap"]
    reasons: list[str] = []
    if not domain:
        reasons.append("missing_domain_gap_summary")
    if domain and config["require_domain_gap_acceptable"]:
        if domain.get("status") != "passed":
            reasons.append("domain_gap_not_passed")
        if domain.get("domain_gap_verdict") != "acceptable_for_next_pilot":
            reasons.append("domain_gap_verdict_not_acceptable")
    return {
        "schema_version": "global-99-real-map-release-domain-gap-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "status": domain.get("status"),
        "domain_gap_verdict": domain.get("domain_gap_verdict"),
    }


def _path_feedback_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    shadow = evidence["shadow"]
    preflight = evidence["preflight"]
    domain = evidence["domain_gap"]
    path_feedback = evidence["path_feedback"]
    reasons: list[str] = []
    context_missing = _max_int("context_id_missing_count", shadow, preflight, domain)
    legacy_fallback = _max_int("legacy_identity_fallback_count", shadow, preflight, domain)
    open_grid = bool(shadow.get("open_grid_fallback_used") or path_feedback.get("open_grid_fallback_used"))
    fallback_count = _max_int("fallback_or_open_grid_count", shadow, preflight, domain)
    if path_feedback and bool(path_feedback.get("open_grid_fallback_used")):
        fallback_count = max(fallback_count, 1)
    regression_count = sum(
        _max_int(field, shadow, preflight, domain)
        for field in (
            "safety_regression_count",
            "contract_violation_count",
            "path_cost_regression_count",
            "risk_regression_count",
            "source_selection_regression_count",
        )
    )
    if config["require_context_ids"] and (context_missing > 0 or legacy_fallback > 0):
        reasons.append("real_map_context_id_missing")
    if config["require_no_open_grid_fallback"] and (open_grid or fallback_count > 0):
        reasons.append("real_map_open_grid_fallback_detected")
    if regression_count > 0:
        reasons.append("real_map_path_feedback_regression")
    if not path_feedback:
        reasons.append("missing_quasi_real_path_feedback_summary")
    return {
        "schema_version": "global-99-real-map-release-path-feedback-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "context_id_missing_count": context_missing,
        "legacy_identity_fallback_count": legacy_fallback,
        "open_grid_fallback_used": open_grid,
        "fallback_or_open_grid_count": fallback_count,
        "safety_regression_count": _max_int("safety_regression_count", shadow, preflight, domain),
        "contract_violation_count": _max_int("contract_violation_count", shadow, preflight, domain),
        "path_cost_regression_count": _max_int("path_cost_regression_count", shadow, preflight, domain),
        "risk_regression_count": _max_int("risk_regression_count", shadow, preflight, domain),
        "source_selection_regression_count": _max_int("source_selection_regression_count", shadow, preflight, domain),
    }


def _scope_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    shadow = evidence["shadow"]
    preflight = evidence["preflight"]
    fallback_rate = _policy_guard_fallback_rate(evidence)
    reasons: list[str] = []
    if fallback_rate > config["max_policy_guard_fallback_rate"]:
        reasons.append("policy_guard_fallback_rate_too_high")
    return {
        "schema_version": "global-99-real-map-release-scope-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "target_coverage_rate": config["target_coverage_rate"],
        "slice_count": _max_int("slice_count", shadow, preflight, evidence["domain_gap"]),
        "roi_group_count": _max_int("roi_group_count", shadow, preflight, evidence["domain_gap"]),
        "policy_guard_fallback_rate": fallback_rate,
        "max_policy_guard_fallback_rate": config["max_policy_guard_fallback_rate"],
    }


def _boundary_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for source_name in ("shadow", "preflight", "domain_gap", "path_feedback"):
        source = evidence[source_name]
        for field in FORBIDDEN_BOOLEAN_FIELDS:
            if source.get(field) is True:
                violations.append({"source": source_name, "field": field, "value": True})
        if _float_default(source.get("canary_traffic_fraction")) > 0:
            violations.append({"source": source_name, "field": "canary_traffic_fraction", "value": source.get("canary_traffic_fraction")})
    reasons = ["real_map_release_boundary_violation"] if violations and config["require_release_boundaries_closed"] else []
    return {
        "schema_version": "global-99-real-map-release-boundary-audit/v1",
        "passed": not reasons,
        "reason_codes": reasons,
        "violations": violations,
        **BOUNDARY_FIELDS,
    }


def _simple_audit(name: str, passed: bool) -> dict[str, Any]:
    return {
        "schema_version": f"global-99-real-map-release-{name}-audit/v1",
        "passed": bool(passed),
        "reason_codes": [] if passed else [f"{name}_audit_blocked"],
    }


def _decision(
    config: dict[str, Any],
    evidence: dict[str, Any],
    lineage: dict[str, Any],
    shadow: dict[str, Any],
    preflight: dict[str, Any],
    domain_gap: dict[str, Any],
    path_feedback: dict[str, Any],
    scope: dict[str, Any],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    reasons = unique_sorted(
        list(lineage["reason_codes"])
        + list(shadow["reason_codes"])
        + list(preflight["reason_codes"])
        + list(domain_gap["reason_codes"])
        + list(path_feedback["reason_codes"])
        + list(scope["reason_codes"])
        + list(boundary["reason_codes"])
    )
    status = "passed" if not reasons else "failed"
    next_required_change = PASS_NEXT_REQUIRED_CHANGE
    verdict = PASS_VERDICT
    if status == "failed":
        verdict = "blocked"
        if any(reason.startswith("missing_real_map_shadow_replay") or reason.startswith("real_map_shadow_replay_not_passed") or reason == "real_map_shadow_replay_wrong_next_required_change" for reason in reasons):
            next_required_change = FIX_SHADOW_NEXT_REQUIRED_CHANGE
        elif any(reason.startswith("missing_real_map_preflight") or reason.startswith("real_map_preflight") for reason in reasons):
            next_required_change = FIX_PREFLIGHT_NEXT_REQUIRED_CHANGE
        elif any(reason.startswith("missing_domain_gap") or reason.startswith("domain_gap") for reason in reasons):
            next_required_change = FIX_DOMAIN_GAP_NEXT_REQUIRED_CHANGE
        elif "real_map_shadow_replay_source_match_failed" in reasons:
            next_required_change = FIX_DETERMINISM_NEXT_REQUIRED_CHANGE
        elif "real_map_context_id_missing" in reasons:
            next_required_change = FIX_CONTEXT_NEXT_REQUIRED_CHANGE
        elif "real_map_open_grid_fallback_detected" in reasons or "real_map_path_feedback_regression" in reasons or "missing_quasi_real_path_feedback_summary" in reasons:
            next_required_change = FIX_PATH_FEEDBACK_NEXT_REQUIRED_CHANGE
        elif "policy_guard_fallback_rate_too_high" in reasons:
            next_required_change = FIX_FALLBACK_NEXT_REQUIRED_CHANGE
        elif "real_map_release_boundary_violation" in reasons:
            next_required_change = BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE
        else:
            next_required_change = FIX_SHADOW_NEXT_REQUIRED_CHANGE
    return {
        "status": status,
        "reason_codes": reasons,
        "real_map_release_governance_verdict": verdict,
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
    lineage: dict[str, Any],
    shadow: dict[str, Any],
    preflight: dict[str, Any],
    domain_gap: dict[str, Any],
    path_feedback: dict[str, Any],
    scope: dict[str, Any],
    boundary: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    shadow_source = evidence["shadow"]
    preflight_source = evidence["preflight"]
    domain_source = evidence["domain_gap"]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_real_map_shadow_replay_status": shadow_source.get("status"),
        "source_real_map_shadow_replay_next_required_change": shadow_source.get("next_required_change"),
        "source_real_map_preflight_status": preflight_source.get("status"),
        "source_real_map_preflight_next_required_change": preflight_source.get("next_required_change"),
        "source_domain_gap_status": domain_source.get("status"),
        "domain_gap_verdict": domain_source.get("domain_gap_verdict"),
        "target_coverage_rate": scope.get("target_coverage_rate"),
        "source_match_audit_passed": shadow["source_match_audit_passed"],
        "scenario_mismatch_count": shadow["scenario_mismatch_count"],
        "max_replay_coverage_delta": shadow["max_replay_coverage_delta"],
        "max_replay_path_cost_delta_m": shadow["max_replay_path_cost_delta_m"],
        "slice_count": scope["slice_count"],
        "roi_group_count": scope["roi_group_count"],
        "context_id_missing_count": path_feedback["context_id_missing_count"],
        "legacy_identity_fallback_count": path_feedback["legacy_identity_fallback_count"],
        "open_grid_fallback_used": path_feedback["open_grid_fallback_used"],
        "fallback_or_open_grid_count": path_feedback["fallback_or_open_grid_count"],
        "safety_regression_count": path_feedback["safety_regression_count"],
        "contract_violation_count": path_feedback["contract_violation_count"],
        "path_cost_regression_count": path_feedback["path_cost_regression_count"],
        "risk_regression_count": path_feedback["risk_regression_count"],
        "source_selection_regression_count": path_feedback["source_selection_regression_count"],
        "policy_guard_fallback_rate": scope["policy_guard_fallback_rate"],
        "real_map_release_governance_preflight_passed": decision["status"] == "passed",
        "real_map_release_governance_verdict": decision["real_map_release_governance_verdict"],
        "evidence_lineage_audit_passed": lineage["passed"],
        "shadow_replay_audit_passed": shadow["passed"],
        "real_map_preflight_audit_passed": preflight["passed"],
        "domain_gap_audit_passed": domain_gap["passed"],
        "path_feedback_audit_passed": path_feedback["passed"],
        "scope_audit_passed": scope["passed"],
        "release_boundary_audit_passed": boundary["passed"],
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


def _manifest(generated_at: str, config_path: Path, output_root: Path, paths: dict[str, Path], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "status": summary["status"],
        "next_required_change": summary["next_required_change"],
    }


def _rejection_report(
    generated_at: str,
    decision: dict[str, Any],
    lineage: dict[str, Any],
    shadow: dict[str, Any],
    preflight: dict[str, Any],
    domain_gap: dict[str, Any],
    path_feedback: dict[str, Any],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "global-99-real-map-release-rejection-report/v1",
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "audit_reason_codes": {
            "lineage": lineage["reason_codes"],
            "shadow_replay": shadow["reason_codes"],
            "real_map_preflight": preflight["reason_codes"],
            "domain_gap": domain_gap["reason_codes"],
            "path_feedback": path_feedback["reason_codes"],
            "boundary": boundary["reason_codes"],
        },
    }


def _render_report(summary: dict[str, Any], rejection: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Global 99 Real Map Release Governance Preflight v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- verdict: `{summary['real_map_release_governance_verdict']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- source_match_audit_passed: `{summary['source_match_audit_passed']}`",
            f"- release_boundary_audit_passed: `{summary['release_boundary_audit_passed']}`",
            "",
            "## Rejections",
            "",
            f"```json\n{json.dumps(rejection, ensure_ascii=False, indent=2)}\n```",
            "",
        ]
    )


def _float_value(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{name} must be numeric")
    return float(value)


def _bool_value(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value


def _float_default(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _max_int(field: str, *sources: dict[str, Any]) -> int:
    return max((_int_value(source.get(field)) for source in sources if source), default=0)


def _policy_guard_fallback_rate(evidence: dict[str, Any]) -> float:
    preflight = evidence["preflight"]
    if isinstance(preflight.get("shadow_replay_policy_guard_fallback_rate"), (int, float)):
        return float(preflight["shadow_replay_policy_guard_fallback_rate"])
    shadow = evidence["shadow"]
    if isinstance(shadow.get("policy_guard_fallback_rate"), (int, float)):
        return float(shadow["policy_guard_fallback_rate"])
    return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
