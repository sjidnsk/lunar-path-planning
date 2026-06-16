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


CONFIG_SCHEMA_VERSION = "global-99-real-map-preflight-config/v1"
SUMMARY_SCHEMA_VERSION = "global-99-real-map-preflight-summary/v1"
MANIFEST_SCHEMA_VERSION = "global-99-real-map-preflight-manifest/v1"
LINEAGE_AUDIT_SCHEMA_VERSION = "global-99-real-map-evidence-lineage-audit/v1"
DOMAIN_GAP_AUDIT_SCHEMA_VERSION = "global-99-real-map-domain-gap-audit/v1"
PATH_FEEDBACK_AUDIT_SCHEMA_VERSION = "global-99-real-map-path-feedback-audit/v1"
SCOPE_AUDIT_SCHEMA_VERSION = "global-99-real-map-scope-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "global-99-real-map-boundary-audit/v1"
KILL_SWITCH_AUDIT_SCHEMA_VERSION = "global-99-real-map-kill-switch-audit/v1"
ROLLBACK_AUDIT_SCHEMA_VERSION = "global-99-real-map-rollback-audit/v1"
TELEMETRY_AUDIT_SCHEMA_VERSION = "global-99-real-map-telemetry-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "global-99-real-map-rejection-report/v1"

DEFAULT_CONFIG = "configs/global_99_real_map_preflight_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_global_99_real_map_preflight_v1"

SHADOW_REPLAY_SUMMARY_FILE = "global-99-shadow-canary-replay-summary.json"
DOMAIN_GAP_SUMMARY_FILE = "quasi-real-map-domain-gap-summary.json"
PATH_FEEDBACK_SUMMARY_FILE = "quasi-real-map-path-feedback-summary.json"
PATH_FEEDBACK_BRIDGE_SUMMARY_FILE = "quasi-real-map-path-feedback-bridge-summary.json"
OPTIONAL_PATH_FEEDBACK_VALIDATION_SUMMARY_FILE = "path-feedback-summary.json"

SUMMARY_FILE = "global-99-real-map-preflight-summary.json"
MANIFEST_FILE = "global-99-real-map-preflight-manifest.json"
LINEAGE_AUDIT_FILE = "global-99-real-map-evidence-lineage-audit.json"
DOMAIN_GAP_AUDIT_FILE = "global-99-real-map-domain-gap-audit.json"
PATH_FEEDBACK_AUDIT_FILE = "global-99-real-map-path-feedback-audit.json"
SCOPE_AUDIT_FILE = "global-99-real-map-scope-audit.json"
BOUNDARY_AUDIT_FILE = "global-99-real-map-boundary-audit.json"
KILL_SWITCH_AUDIT_FILE = "global-99-real-map-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "global-99-real-map-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "global-99-real-map-telemetry-audit.json"
REJECTION_REPORT_FILE = "global-99-real-map-rejection-report.json"
REPORT_FILE = "global-99-real-map-preflight-report.md"

PASS_VERDICT = "eligible_for_global_99_real_map_shadow_replay"
PASS_NEXT_REQUIRED_CHANGE = "global_99_real_map_shadow_replay"
FIX_SHADOW_REPLAY_NEXT_REQUIRED_CHANGE = "fix_global_99_shadow_canary_replay"
FIX_DOMAIN_GAP_NEXT_REQUIRED_CHANGE = "fix_quasi_real_map_domain_gap_evidence"
EXPAND_REAL_MAP_NEXT_REQUIRED_CHANGE = "expand_real_map_roi_coverage"
FIX_CONTEXT_NEXT_REQUIRED_CHANGE = "fix_real_map_context_identity"
FIX_PATH_FEEDBACK_NEXT_REQUIRED_CHANGE = "fix_real_map_path_feedback_contract"
FIX_FALLBACK_NEXT_REQUIRED_CHANGE = "fix_global_99_shadow_canary_guard_fallback"
BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE = "resolve_global_99_real_map_preflight_boundary_rejections"

BOUNDARY_FIELDS = {
    "real_map_preflight_only": True,
    "uses_lola_quasi_real_roi": True,
    "uses_path_feedback_sidecar": True,
    "uses_npz_or_sidecar": False,
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
    "uses_path_planner": False,
}

FORBIDDEN_BOUNDARY_FIELDS = (
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
    "uses_path_planner",
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
    parser = argparse.ArgumentParser(description="Run Global 99 Real Map Preflight v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_global_99_real_map_preflight(
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
                "real_map_preflight_verdict": summary["real_map_preflight_verdict"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_global_99_real_map_preflight(
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
    domain_gap = _domain_gap_audit(config, evidence)
    path_feedback = _path_feedback_audit(config, evidence)
    scope = _scope_audit(config, evidence)
    boundary = _boundary_audit(config, evidence)
    kill_switch = _kill_switch_audit(lineage, boundary)
    rollback = _rollback_audit(lineage, boundary)
    telemetry = _telemetry_audit(evidence, domain_gap, path_feedback)
    decision = _decision(
        config,
        evidence,
        lineage,
        domain_gap,
        path_feedback,
        scope,
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
        lineage=lineage,
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
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)
    rejection_report = _rejection_report(generated_at, decision, lineage, domain_gap, path_feedback, scope, boundary)

    for key, payload in (
        ("lineage", lineage),
        ("domain_gap", domain_gap),
        ("path_feedback", path_feedback),
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
    for key in (
        "source_shadow_canary_replay_root",
        "source_quasi_real_domain_gap_root",
        "source_path_feedback_validation_root",
    ):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    normalized = dict(payload)
    normalized["source_shadow_canary_replay_root"] = str(
        resolve_path(Path(payload["source_shadow_canary_replay_root"]), repo_root)
    )
    normalized["source_quasi_real_domain_gap_root"] = str(
        resolve_path(Path(payload["source_quasi_real_domain_gap_root"]), repo_root)
    )
    normalized["source_path_feedback_validation_root"] = str(
        resolve_path(Path(payload["source_path_feedback_validation_root"]), repo_root)
    )
    normalized["target_coverage_rate"] = _bounded_float(payload.get("target_coverage_rate"), "target_coverage_rate")
    normalized["min_real_map_slice_count"] = _non_negative_int(
        payload.get("min_real_map_slice_count"),
        "min_real_map_slice_count",
    )
    normalized["min_real_map_roi_group_count"] = _non_negative_int(
        payload.get("min_real_map_roi_group_count"),
        "min_real_map_roi_group_count",
    )
    normalized["max_policy_guard_fallback_rate"] = _bounded_float(
        payload.get("max_policy_guard_fallback_rate"),
        "max_policy_guard_fallback_rate",
    )
    normalized["require_domain_gap_acceptable"] = _bool_value(
        payload.get("require_domain_gap_acceptable"),
        "require_domain_gap_acceptable",
    )
    normalized["require_no_open_grid_fallback"] = _bool_value(
        payload.get("require_no_open_grid_fallback"),
        "require_no_open_grid_fallback",
    )
    normalized["require_context_ids"] = _bool_value(payload.get("require_context_ids"), "require_context_ids")
    normalized["real_map_preflight_only"] = _bool_value(
        payload.get("real_map_preflight_only"),
        "real_map_preflight_only",
    )
    normalized["canary_traffic_fraction"] = _fraction_float(
        payload.get("canary_traffic_fraction"),
        "canary_traffic_fraction",
    )
    return normalized


def _source_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    shadow_replay_root = resolve_path(Path(config["source_shadow_canary_replay_root"]), repo_root)
    domain_gap_root = resolve_path(Path(config["source_quasi_real_domain_gap_root"]), repo_root)
    path_feedback_root = resolve_path(Path(config["source_path_feedback_validation_root"]), repo_root)
    return {
        "shadow_canary_replay_root": shadow_replay_root,
        "quasi_real_domain_gap_root": domain_gap_root,
        "path_feedback_validation_root": path_feedback_root,
        "shadow_canary_replay_summary": shadow_replay_root / SHADOW_REPLAY_SUMMARY_FILE,
        "quasi_real_domain_gap_summary": domain_gap_root / DOMAIN_GAP_SUMMARY_FILE,
        "quasi_real_path_feedback_summary": domain_gap_root / PATH_FEEDBACK_SUMMARY_FILE,
        "quasi_real_path_feedback_bridge_summary": domain_gap_root / PATH_FEEDBACK_BRIDGE_SUMMARY_FILE,
        "optional_path_feedback_validation_summary": path_feedback_root / OPTIONAL_PATH_FEEDBACK_VALIDATION_SUMMARY_FILE,
    }


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "lineage": output_root / LINEAGE_AUDIT_FILE,
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
    read_reasons: list[str] = []
    optional_read_reasons: list[str] = []
    return {
        "shadow_canary_replay": _read_json(
            source_paths["shadow_canary_replay_summary"],
            read_reasons,
            "shadow_canary_replay_summary",
        ),
        "domain_gap": _read_json(
            source_paths["quasi_real_domain_gap_summary"],
            read_reasons,
            "quasi_real_domain_gap_summary",
        ),
        "path_feedback": _read_json(
            source_paths["quasi_real_path_feedback_summary"],
            read_reasons,
            "quasi_real_path_feedback_summary",
        ),
        "path_feedback_bridge": _read_json(
            source_paths["quasi_real_path_feedback_bridge_summary"],
            read_reasons,
            "quasi_real_path_feedback_bridge_summary",
        ),
        "optional_path_feedback_validation": _read_optional_json(
            source_paths["optional_path_feedback_validation_summary"],
            optional_read_reasons,
            "optional_path_feedback_validation_summary",
        ),
        "read_reason_codes": unique_sorted(read_reasons),
        "optional_read_reason_codes": unique_sorted(optional_read_reasons),
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


def _read_optional_json(path: Path, reasons: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(f"missing_{label}")
        return {}
    return _read_json(path, reasons, label)


def _lineage_audit(source_paths: dict[str, Path], evidence: dict[str, Any]) -> dict[str, Any]:
    reasons = list(evidence["read_reason_codes"])
    replay = evidence["shadow_canary_replay"]
    if replay and replay.get("status") != "passed":
        reasons.append("shadow_canary_replay_not_passed")
    if replay and replay.get("next_required_change") != "global_99_real_map_preflight":
        reasons.append("shadow_canary_replay_next_required_change_invalid")
    if replay and replay.get("source_match_audit_passed") is not True:
        reasons.append("shadow_replay_source_match_not_passed")
    return {
        "schema_version": LINEAGE_AUDIT_SCHEMA_VERSION,
        "evidence_lineage_audit_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "source_paths": {key: str(value) for key, value in source_paths.items()},
    }


def _domain_gap_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    domain = evidence["domain_gap"]
    bridge = evidence["path_feedback_bridge"]
    reasons: list[str] = []
    if not domain:
        reasons.append("domain_gap_evidence_missing")
    elif domain.get("status") != "passed":
        reasons.append("domain_gap_not_passed")
    if config["require_domain_gap_acceptable"] and domain.get("domain_gap_verdict") != "acceptable_for_next_pilot":
        reasons.append("domain_gap_verdict_not_acceptable")

    slice_count = _int_from(domain.get("slice_count"), 0)
    roi_group_count = _int_from(domain.get("roi_group_count"), 0)
    context_missing = _max_int(domain.get("context_id_missing_count"), bridge.get("context_id_missing_count"))
    legacy_fallback = _max_int(domain.get("legacy_identity_fallback_count"), bridge.get("legacy_identity_fallback_count"))
    fallback_or_open_grid = _int_from(domain.get("fallback_or_open_grid_count"), 0)
    safety_regression = _int_from(domain.get("safety_regression_count"), 0)
    contract_violation = _int_from(domain.get("contract_violation_count"), 0)
    path_cost_regression = _int_from(domain.get("path_cost_regression_count"), 0)
    risk_regression = _int_from(domain.get("risk_regression_count"), 0)
    source_selection_regression = _int_from(domain.get("source_selection_regression_count"), 0)

    if slice_count < int(config["min_real_map_slice_count"]):
        reasons.append("real_map_slice_count_below_minimum")
    if roi_group_count < int(config["min_real_map_roi_group_count"]):
        reasons.append("real_map_roi_group_count_below_minimum")
    if config["require_context_ids"] and context_missing > 0:
        reasons.append("real_map_context_id_missing")
    if config["require_context_ids"] and legacy_fallback > 0:
        reasons.append("real_map_legacy_identity_fallback_used")
    if fallback_or_open_grid > 0:
        reasons.append("real_map_open_grid_fallback_detected")
    if safety_regression + path_cost_regression + risk_regression + source_selection_regression > 0:
        reasons.append("real_map_path_feedback_regression")
    if contract_violation > 0:
        reasons.append("real_map_contract_violation")

    return {
        "schema_version": DOMAIN_GAP_AUDIT_SCHEMA_VERSION,
        "domain_gap_audit_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "domain_gap_verdict": domain.get("domain_gap_verdict", "missing"),
        "slice_count": slice_count,
        "roi_group_count": roi_group_count,
        "context_id_missing_count": context_missing,
        "legacy_identity_fallback_count": legacy_fallback,
        "fallback_or_open_grid_count": fallback_or_open_grid,
        "safety_regression_count": safety_regression,
        "contract_violation_count": contract_violation,
        "path_cost_regression_count": path_cost_regression,
        "risk_regression_count": risk_regression,
        "source_selection_regression_count": source_selection_regression,
        "min_real_map_slice_count": int(config["min_real_map_slice_count"]),
        "min_real_map_roi_group_count": int(config["min_real_map_roi_group_count"]),
    }


def _path_feedback_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    path_feedback = evidence["path_feedback"]
    bridge = evidence["path_feedback_bridge"]
    validation = evidence["optional_path_feedback_validation"]
    reasons: list[str] = []
    if not path_feedback:
        reasons.append("missing_quasi_real_path_feedback_summary")
    if not bridge:
        reasons.append("missing_quasi_real_path_feedback_bridge_summary")
    open_grid_sources = []
    for source_name, payload in (
        ("quasi_real_path_feedback", path_feedback),
        ("optional_path_feedback_validation", validation),
    ):
        if payload.get("open_grid_fallback_used") is True:
            open_grid_sources.append(source_name)
    gate = path_feedback.get("open_grid_fallback_used_gate", {})
    if isinstance(gate, dict) and gate.get("actual") is True:
        open_grid_sources.append("quasi_real_path_feedback_gate")
    if config["require_no_open_grid_fallback"] and open_grid_sources:
        reasons.append("real_map_open_grid_fallback_detected")
    if isinstance(gate, dict) and gate and gate.get("status") not in (None, "passed"):
        reasons.append("real_map_path_feedback_regression")
    return {
        "schema_version": PATH_FEEDBACK_AUDIT_SCHEMA_VERSION,
        "path_feedback_audit_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "quasi_real_path_feedback_present": bool(path_feedback),
        "path_feedback_bridge_present": bool(bridge),
        "optional_path_feedback_validation_present": bool(validation),
        "optional_read_reason_codes": evidence["optional_read_reason_codes"],
        "scenario_count": _int_from(path_feedback.get("scenario_count"), 0),
        "open_grid_fallback_sources": sorted(set(open_grid_sources)),
        "require_no_open_grid_fallback": config["require_no_open_grid_fallback"],
    }


def _scope_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    replay = evidence["shadow_canary_replay"]
    reasons: list[str] = []
    if not config["real_map_preflight_only"]:
        reasons.append("real_map_preflight_only_disabled")
    if float(config["canary_traffic_fraction"]) != 0.0:
        reasons.append("canary_traffic_fraction_must_be_zero")
    if replay.get("status") != "passed":
        reasons.append("shadow_canary_replay_not_passed")
    if replay.get("shadow_replay_passed") is not True or replay.get("offline_canary_replay_passed") is not True:
        reasons.append("shadow_canary_replay_not_complete")
    if replay.get("source_match_audit_passed") is not True:
        reasons.append("shadow_replay_source_match_not_passed")
    fallback_rate = float(replay.get("policy_guard_fallback_rate", 1.0))
    if fallback_rate > float(config["max_policy_guard_fallback_rate"]):
        reasons.append("shadow_replay_policy_guard_fallback_rate_too_high")
    return {
        "schema_version": SCOPE_AUDIT_SCHEMA_VERSION,
        "scope_audit_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "target_coverage_rate": float(config["target_coverage_rate"]),
        "max_policy_guard_fallback_rate": float(config["max_policy_guard_fallback_rate"]),
        "shadow_replay_policy_guard_fallback_rate": fallback_rate,
        "real_map_preflight_only": config["real_map_preflight_only"],
        "canary_traffic_fraction": 0.0,
    }


def _boundary_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    violations = []
    for source_name in ("shadow_canary_replay", "domain_gap", "path_feedback", "path_feedback_bridge"):
        payload = evidence[source_name]
        for field in FORBIDDEN_BOUNDARY_FIELDS:
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
        "reason_codes": [] if not violations else ["global_99_real_map_boundary_violation"],
        "violations": violations,
        **BOUNDARY_FIELDS,
    }


def _kill_switch_audit(lineage: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    passed = lineage["evidence_lineage_audit_passed"] and boundary["boundary_audit_passed"]
    return {
        "schema_version": KILL_SWITCH_AUDIT_SCHEMA_VERSION,
        "kill_switch_audit_passed": passed,
        "reason_codes": [] if passed else ["kill_switch_contract_blocked"],
        "kill_switch_required_before_real_map_shadow_replay": True,
        "kill_switch_real_control_plane_connected": False,
    }


def _rollback_audit(lineage: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    passed = lineage["evidence_lineage_audit_passed"] and boundary["boundary_audit_passed"]
    return {
        "schema_version": ROLLBACK_AUDIT_SCHEMA_VERSION,
        "rollback_audit_passed": passed,
        "reason_codes": [] if passed else ["rollback_contract_blocked"],
        "rollback_required_before_real_map_shadow_replay": True,
        "rollback_real_control_plane_connected": False,
    }


def _telemetry_audit(
    evidence: dict[str, Any],
    domain_gap: dict[str, Any],
    path_feedback: dict[str, Any],
) -> dict[str, Any]:
    replay = evidence["shadow_canary_replay"]
    passed = (
        replay.get("status") == "passed"
        and domain_gap["slice_count"] > 0
        and domain_gap["roi_group_count"] > 0
        and path_feedback["quasi_real_path_feedback_present"]
        and path_feedback["path_feedback_bridge_present"]
    )
    return {
        "schema_version": TELEMETRY_AUDIT_SCHEMA_VERSION,
        "telemetry_audit_passed": passed,
        "reason_codes": [] if passed else ["telemetry_evidence_incomplete"],
        "tracked_counters": {
            "slice_count": domain_gap["slice_count"],
            "roi_group_count": domain_gap["roi_group_count"],
            "path_feedback_scenario_count": path_feedback["scenario_count"],
            "shadow_replay_policy_guard_fallback_rate": float(replay.get("policy_guard_fallback_rate", 1.0)),
        },
    }


def _decision(
    config: dict[str, Any],
    evidence: dict[str, Any],
    lineage: dict[str, Any],
    domain_gap: dict[str, Any],
    path_feedback: dict[str, Any],
    scope: dict[str, Any],
    boundary: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    reason_codes = unique_sorted(
        list(lineage["reason_codes"])
        + list(domain_gap["reason_codes"])
        + list(path_feedback["reason_codes"])
        + list(scope["reason_codes"])
        + list(boundary["reason_codes"])
        + list(kill_switch["reason_codes"])
        + list(rollback["reason_codes"])
        + list(telemetry["reason_codes"])
    )
    read_reasons = set(evidence["read_reason_codes"])
    replay = evidence["shadow_canary_replay"]
    domain = evidence["domain_gap"]

    if any(reason for reason in read_reasons if "shadow_canary_replay" in reason):
        next_required_change = FIX_SHADOW_REPLAY_NEXT_REQUIRED_CHANGE
    elif replay.get("status") != "passed" or replay.get("next_required_change") != "global_99_real_map_preflight":
        next_required_change = FIX_SHADOW_REPLAY_NEXT_REQUIRED_CHANGE
    elif replay.get("source_match_audit_passed") is not True:
        next_required_change = FIX_SHADOW_REPLAY_NEXT_REQUIRED_CHANGE
    elif any(reason for reason in read_reasons if "domain_gap" in reason):
        next_required_change = FIX_DOMAIN_GAP_NEXT_REQUIRED_CHANGE
    elif domain.get("status") != "passed" or (
        config["require_domain_gap_acceptable"] and domain.get("domain_gap_verdict") != "acceptable_for_next_pilot"
    ):
        next_required_change = FIX_DOMAIN_GAP_NEXT_REQUIRED_CHANGE
    elif "real_map_slice_count_below_minimum" in reason_codes or "real_map_roi_group_count_below_minimum" in reason_codes:
        next_required_change = EXPAND_REAL_MAP_NEXT_REQUIRED_CHANGE
    elif "real_map_context_id_missing" in reason_codes or "real_map_legacy_identity_fallback_used" in reason_codes:
        next_required_change = FIX_CONTEXT_NEXT_REQUIRED_CHANGE
    elif (
        "real_map_open_grid_fallback_detected" in reason_codes
        or "real_map_path_feedback_regression" in reason_codes
        or "real_map_contract_violation" in reason_codes
        or "missing_quasi_real_path_feedback_summary" in reason_codes
        or "missing_quasi_real_path_feedback_bridge_summary" in reason_codes
    ):
        next_required_change = FIX_PATH_FEEDBACK_NEXT_REQUIRED_CHANGE
    elif "shadow_replay_policy_guard_fallback_rate_too_high" in reason_codes:
        next_required_change = FIX_FALLBACK_NEXT_REQUIRED_CHANGE
    elif not boundary["boundary_audit_passed"]:
        next_required_change = BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE
    elif reason_codes:
        next_required_change = "resolve_global_99_real_map_preflight_rejections"
    else:
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    passed = not reason_codes and next_required_change == PASS_NEXT_REQUIRED_CHANGE
    return {
        "status": "passed" if passed else "failed",
        "reason_codes": reason_codes,
        "real_map_preflight_passed": passed,
        "real_map_preflight_verdict": PASS_VERDICT if passed else "resolve_global_99_real_map_preflight_rejections",
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
    replay = evidence["shadow_canary_replay"]
    domain = evidence["domain_gap"]
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
        "domain_gap_audit": str(paths["domain_gap"]),
        "path_feedback_audit": str(paths["path_feedback"]),
        "scope_audit": str(paths["scope"]),
        "boundary_audit": str(paths["boundary"]),
        "kill_switch_audit": str(paths["kill_switch"]),
        "rollback_audit": str(paths["rollback"]),
        "telemetry_audit": str(paths["telemetry"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "source_shadow_canary_replay_status": replay.get("status", "missing"),
        "source_shadow_canary_replay_next_required_change": replay.get("next_required_change", "missing"),
        "source_domain_gap_status": domain.get("status", "missing"),
        "target_coverage_rate": scope["target_coverage_rate"],
        "domain_gap_verdict": domain_gap["domain_gap_verdict"],
        "slice_count": domain_gap["slice_count"],
        "roi_group_count": domain_gap["roi_group_count"],
        "context_id_missing_count": domain_gap["context_id_missing_count"],
        "legacy_identity_fallback_count": domain_gap["legacy_identity_fallback_count"],
        "fallback_or_open_grid_count": domain_gap["fallback_or_open_grid_count"],
        "safety_regression_count": domain_gap["safety_regression_count"],
        "contract_violation_count": domain_gap["contract_violation_count"],
        "path_cost_regression_count": domain_gap["path_cost_regression_count"],
        "risk_regression_count": domain_gap["risk_regression_count"],
        "source_selection_regression_count": domain_gap["source_selection_regression_count"],
        "shadow_replay_source_match_audit_passed": bool(replay.get("source_match_audit_passed", False)),
        "shadow_replay_policy_guard_fallback_rate": scope["shadow_replay_policy_guard_fallback_rate"],
        "real_map_preflight_passed": decision["real_map_preflight_passed"],
        "real_map_preflight_verdict": decision["real_map_preflight_verdict"],
        "evidence_lineage_audit_passed": lineage["evidence_lineage_audit_passed"],
        "domain_gap_audit_passed": domain_gap["domain_gap_audit_passed"],
        "path_feedback_audit_passed": path_feedback["path_feedback_audit_passed"],
        "scope_audit_passed": scope["scope_audit_passed"],
        "boundary_audit_passed": boundary["boundary_audit_passed"],
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
        "stage": "Global 99 Real Map Preflight v1",
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "summary_status": summary["status"],
        "next_required_change": summary["next_required_change"],
        "boundary_fields": BOUNDARY_FIELDS,
    }


def _rejection_report(
    generated_at: str,
    decision: dict[str, Any],
    lineage: dict[str, Any],
    domain_gap: dict[str, Any],
    path_feedback: dict[str, Any],
    scope: dict[str, Any],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "lineage_reason_codes": lineage["reason_codes"],
        "domain_gap_reason_codes": domain_gap["reason_codes"],
        "path_feedback_reason_codes": path_feedback["reason_codes"],
        "scope_reason_codes": scope["reason_codes"],
        "boundary_reason_codes": boundary["reason_codes"],
        "boundary_violations": boundary["violations"],
        "release_allowed": False,
        "default_policy_replacement_allowed": False,
        "real_executor_connection_allowed": False,
        "online_canary_allowed": False,
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Global 99 Real Map Preflight v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- verdict: `{summary['real_map_preflight_verdict']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- domain_gap_verdict: `{summary['domain_gap_verdict']}`",
            f"- slice_count: `{summary['slice_count']}`",
            f"- roi_group_count: `{summary['roi_group_count']}`",
            f"- fallback_or_open_grid_count: `{summary['fallback_or_open_grid_count']}`",
            f"- boundary_audit_passed: `{summary['boundary_audit_passed']}`",
            "",
            "This stage audits existing LOLA quasi-real ROI, path-feedback sidecar, and synthetic shadow replay evidence. "
            "It does not publish a checkpoint, replace the default policy, connect a real executor, start online canary traffic, "
            "run PPO, or modify network/action-space/default-A* behavior.",
            "",
            "## Rejections",
            "",
            json.dumps(rejection_report, ensure_ascii=False, indent=2, sort_keys=True),
            "",
        ]
    )


def _bounded_float(value: Any, name: str) -> float:
    result = _float_value(value, name)
    if result <= 0.0 or result > 1.0:
        raise ConfigError(f"{name} must be > 0 and <= 1")
    return result


def _fraction_float(value: Any, name: str) -> float:
    result = _float_value(value, name)
    if result < 0.0 or result > 1.0:
        raise ConfigError(f"{name} must be >= 0 and <= 1")
    return result


def _float_value(value: Any, name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ConfigError(f"{name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ConfigError(f"{name} must be an integer >= 0")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer >= 0") from exc
    if result < 0:
        raise ConfigError(f"{name} must be an integer >= 0")
    return result


def _bool_value(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value


def _int_from(value: Any, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _max_int(*values: Any) -> int:
    return max(_int_from(value, 0) for value in values)


if __name__ == "__main__":
    raise SystemExit(main())
