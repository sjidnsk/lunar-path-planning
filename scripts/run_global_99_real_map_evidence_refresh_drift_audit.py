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


CONFIG_SCHEMA_VERSION = "global-99-real-map-evidence-refresh-drift-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "global-99-real-map-evidence-refresh-drift-audit-summary/v1"
DEFAULT_CONFIG = "configs/global_99_real_map_evidence_refresh_drift_audit_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_global_99_real_map_evidence_refresh_drift_audit_v1"

SHADOW_CANARY_SUMMARY_FILE = "global-99-real-map-shadow-canary-replay-summary.json"
SHADOW_REPLAY_SUMMARY_FILE = "global-99-real-map-shadow-replay-summary.json"
DOMAIN_GAP_SUMMARY_FILE = "quasi-real-map-domain-gap-summary.json"
PATH_FEEDBACK_SUMMARY_FILE = "quasi-real-map-path-feedback-summary.json"
PATH_FEEDBACK_BRIDGE_SUMMARY_FILE = "quasi-real-map-path-feedback-bridge-summary.json"
PATH_FEEDBACK_MANIFEST_FILE = "quasi-real-map-path-feedback-manifest.json"
SLICES_FILE = "quasi-real-map-slices.jsonl"

SUMMARY_FILE = "global-99-real-map-evidence-refresh-drift-audit-summary.json"
MANIFEST_FILE = "global-99-real-map-evidence-refresh-drift-audit-manifest.json"
LINEAGE_AUDIT_FILE = "global-99-real-map-evidence-lineage-audit.json"
FINGERPRINT_AUDIT_FILE = "global-99-real-map-evidence-fingerprint-audit.json"
MANIFEST_SIDECAR_AUDIT_FILE = "global-99-real-map-manifest-sidecar-audit.json"
CONTEXT_AUDIT_FILE = "global-99-real-map-context-audit.json"
SOURCE_MATCH_AUDIT_FILE = "global-99-real-map-source-match-drift-audit.json"
BOUNDARY_AUDIT_FILE = "global-99-real-map-evidence-refresh-drift-boundary-audit.json"
KILL_SWITCH_AUDIT_FILE = "global-99-real-map-evidence-refresh-drift-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "global-99-real-map-evidence-refresh-drift-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "global-99-real-map-evidence-refresh-drift-telemetry-audit.json"
REJECTION_REPORT_FILE = "global-99-real-map-evidence-refresh-drift-rejection-report.json"
REPORT_FILE = "global-99-real-map-evidence-refresh-drift-audit-report.md"

PASS_VERDICT = "eligible_for_global_99_real_map_multi_roi_generalization"
PASS_NEXT_REQUIRED_CHANGE = "global_99_real_map_multi_roi_generalization"
FIX_SHADOW_CANARY_NEXT_REQUIRED_CHANGE = "fix_global_99_real_map_shadow_canary_replay"
FIX_DOMAIN_GAP_NEXT_REQUIRED_CHANGE = "fix_quasi_real_map_domain_gap_evidence"
FIX_DRIFT_NEXT_REQUIRED_CHANGE = "fix_global_99_real_map_evidence_drift"
FIX_CONTEXT_NEXT_REQUIRED_CHANGE = "fix_real_map_context_identity"
FIX_PATH_FEEDBACK_NEXT_REQUIRED_CHANGE = "fix_real_map_path_feedback_contract"
FIX_DETERMINISM_NEXT_REQUIRED_CHANGE = "fix_global_99_real_map_shadow_canary_replay_determinism"
FIX_FALLBACK_NEXT_REQUIRED_CHANGE = "fix_global_99_real_map_shadow_canary_guard_fallback"
BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE = "resolve_global_99_real_map_evidence_refresh_boundary_rejections"

BOUNDARY_FIELDS = {
    "evidence_refresh_only": True,
    "uses_lola_quasi_real_roi": True,
    "uses_path_feedback_sidecar": True,
    "uses_path_planner": False,
    "canary_traffic_fraction": 0.0,
    "starts_online_canary": False,
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
    parser = argparse.ArgumentParser(description="Run Global 99 Real Map Evidence Refresh / Drift Audit v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_global_99_real_map_evidence_refresh_drift_audit(
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
        "evidence_refresh_drift_verdict": summary["evidence_refresh_drift_verdict"],
        "next_required_change": summary["next_required_change"],
        "summary": summary["summary"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_global_99_real_map_evidence_refresh_drift_audit(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_config(Path(config_path), repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    source_paths = _source_paths(config, repo_root)
    evidence = _load_evidence(source_paths)

    lineage = _lineage_audit(config, evidence)
    manifest_sidecar = _manifest_sidecar_audit(config, evidence, repo_root)
    context = _context_audit(config, evidence)
    source_match = _source_match_drift_audit(config, evidence)
    fingerprint = _fingerprint_audit(source_paths, manifest_sidecar)
    boundary = _boundary_audit(config, evidence)
    kill_switch = _simple_audit("kill-switch", boundary["passed"] and lineage["passed"])
    rollback = _simple_audit("rollback", boundary["passed"] and source_match["passed"])
    telemetry = _simple_audit("telemetry", boundary["passed"] and manifest_sidecar["passed"])
    decision = _decision(lineage, fingerprint, manifest_sidecar, context, source_match, boundary)
    generated_at = utc_now()

    summary = _summary(
        generated_at=generated_at,
        config_path=Path(config_path),
        output_root=output_root,
        paths=paths,
        source_paths=source_paths,
        evidence=evidence,
        lineage=lineage,
        fingerprint=fingerprint,
        manifest_sidecar=manifest_sidecar,
        context=context,
        source_match=source_match,
        boundary=boundary,
        kill_switch=kill_switch,
        rollback=rollback,
        telemetry=telemetry,
        decision=decision,
        repo_root=Path(repo_root),
    )
    manifest = {
        "schema_version": "global-99-real-map-evidence-refresh-drift-audit-manifest/v1",
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "artifacts": {k: str(v) for k, v in paths.items()},
        "status": summary["status"],
        "next_required_change": summary["next_required_change"],
    }
    rejection = {
        "schema_version": "global-99-real-map-evidence-refresh-drift-rejection-report/v1",
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
    }
    for key, payload in (
        ("lineage", lineage),
        ("fingerprint", fingerprint),
        ("manifest_sidecar", manifest_sidecar),
        ("context", context),
        ("source_match", source_match),
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
    for key in (
        "source_real_map_shadow_canary_replay_root",
        "source_quasi_real_domain_gap_root",
        "source_real_map_shadow_replay_root",
    ):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(payload[key]), repo_root))
    for key in ("min_slice_count", "min_roi_group_count"):
        if not isinstance(payload.get(key), int) or isinstance(payload.get(key), bool):
            raise ConfigError(f"{key} must be an integer")
    for key in ("source_match_coverage_tolerance", "source_match_path_cost_tolerance_m", "max_policy_guard_fallback_rate", "canary_traffic_fraction"):
        normalized[key] = _float_value(payload.get(key), key)
    for key in (
        "require_shadow_canary_replay_passed",
        "require_domain_gap_acceptable",
        "require_context_ids",
        "require_no_legacy_identity_fallback",
        "require_no_open_grid_fallback",
        "require_source_match_audit_passed",
        "require_all_contract_and_sidecar_paths",
        "evidence_refresh_only",
    ):
        if not isinstance(payload.get(key), bool):
            raise ConfigError(f"{key} must be a boolean")
    return normalized


def _source_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    shadow_canary_root = resolve_path(Path(config["source_real_map_shadow_canary_replay_root"]), repo_root)
    domain_gap_root = resolve_path(Path(config["source_quasi_real_domain_gap_root"]), repo_root)
    shadow_replay_root = resolve_path(Path(config["source_real_map_shadow_replay_root"]), repo_root)
    return {
        "real_map_shadow_canary_replay_root": shadow_canary_root,
        "quasi_real_domain_gap_root": domain_gap_root,
        "real_map_shadow_replay_root": shadow_replay_root,
        "real_map_shadow_canary_replay_summary": shadow_canary_root / SHADOW_CANARY_SUMMARY_FILE,
        "real_map_shadow_replay_summary": shadow_replay_root / SHADOW_REPLAY_SUMMARY_FILE,
        "domain_gap_summary": domain_gap_root / DOMAIN_GAP_SUMMARY_FILE,
        "path_feedback_summary": domain_gap_root / PATH_FEEDBACK_SUMMARY_FILE,
        "path_feedback_bridge_summary": domain_gap_root / PATH_FEEDBACK_BRIDGE_SUMMARY_FILE,
        "path_feedback_manifest": domain_gap_root / PATH_FEEDBACK_MANIFEST_FILE,
        "slices": domain_gap_root / SLICES_FILE,
    }


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "lineage": output_root / LINEAGE_AUDIT_FILE,
        "fingerprint": output_root / FINGERPRINT_AUDIT_FILE,
        "manifest_sidecar": output_root / MANIFEST_SIDECAR_AUDIT_FILE,
        "context": output_root / CONTEXT_AUDIT_FILE,
        "source_match": output_root / SOURCE_MATCH_AUDIT_FILE,
        "boundary": output_root / BOUNDARY_AUDIT_FILE,
        "kill_switch": output_root / KILL_SWITCH_AUDIT_FILE,
        "rollback": output_root / ROLLBACK_AUDIT_FILE,
        "telemetry": output_root / TELEMETRY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _load_evidence(source_paths: dict[str, Path]) -> dict[str, Any]:
    reasons: list[str] = []
    slices = _read_jsonl(source_paths["slices"], reasons, "slices")
    return {
        "shadow_canary": _read_json(source_paths["real_map_shadow_canary_replay_summary"], reasons, "real_map_shadow_canary_replay"),
        "shadow_replay": _read_json(source_paths["real_map_shadow_replay_summary"], reasons, "real_map_shadow_replay"),
        "domain_gap": _read_json(source_paths["domain_gap_summary"], reasons, "domain_gap"),
        "path_feedback": _read_json(source_paths["path_feedback_summary"], reasons, "path_feedback"),
        "bridge": _read_json(source_paths["path_feedback_bridge_summary"], reasons, "path_feedback_bridge"),
        "manifest": _read_json(source_paths["path_feedback_manifest"], reasons, "path_feedback_manifest"),
        "slices": slices,
        "read_reason_codes": unique_sorted(reasons),
    }


def _read_json(path: Path, reasons: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(f"missing_{label}_summary" if label.endswith(("replay", "gap", "feedback", "bridge")) else f"missing_{label}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        reasons.append(f"invalid_{label}")
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, reasons: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        reasons.append(f"missing_{label}")
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            reasons.append(f"invalid_{label}")
            return []
        if not isinstance(payload, dict):
            reasons.append(f"invalid_{label}")
            return []
        rows.append(payload)
    return rows


def _lineage_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    shadow = evidence["shadow_canary"]
    domain = evidence["domain_gap"]
    path_feedback = evidence["path_feedback"]
    reasons = list(evidence["read_reason_codes"])
    if shadow and config["require_shadow_canary_replay_passed"] and shadow.get("status") != "passed":
        reasons.append("real_map_shadow_canary_replay_not_passed")
    if shadow and shadow.get("next_required_change") != "global_99_real_map_evidence_refresh_drift_audit":
        reasons.append("real_map_shadow_canary_replay_wrong_next_required_change")
    if domain and config["require_domain_gap_acceptable"]:
        if domain.get("status") != "passed" or domain.get("domain_gap_verdict") != "acceptable_for_next_pilot":
            reasons.append("domain_gap_evidence_not_acceptable")
    slice_count = _int_value(domain.get("slice_count")) or len(evidence["slices"])
    roi_group_count = _int_value(domain.get("roi_group_count")) or len({row.get("roi_name") or row.get("scenario_group") for row in evidence["slices"] if row})
    if slice_count < config["min_slice_count"] or roi_group_count < config["min_roi_group_count"]:
        reasons.append("real_map_evidence_scope_insufficient")
    path_feedback_regression_count = max(
        _int_value(domain.get("fallback_or_open_grid_count")),
        _int_value(domain.get("open_grid_fallback_count")),
        int(bool(path_feedback.get("open_grid_fallback_used"))),
        _int_value(domain.get("safety_regression_count")),
        _int_value(domain.get("contract_violation_count")),
        _int_value(domain.get("path_cost_regression_count")),
        _int_value(domain.get("risk_regression_count")),
        _int_value(domain.get("source_selection_regression_count")),
    )
    if config["require_no_open_grid_fallback"] and path_feedback_regression_count:
        reasons.append("real_map_path_feedback_contract_incomplete")
    return {
        "schema_version": "global-99-real-map-evidence-lineage-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "source_real_map_shadow_canary_replay_status": shadow.get("status"),
        "source_real_map_shadow_canary_replay_next_required_change": shadow.get("next_required_change"),
        "source_domain_gap_status": domain.get("status"),
        "domain_gap_verdict": domain.get("domain_gap_verdict"),
        "slice_count": slice_count,
        "roi_group_count": roi_group_count,
        "path_feedback_regression_or_fallback_count": path_feedback_regression_count,
    }


def _manifest_sidecar_audit(config: dict[str, Any], evidence: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    reasons: list[str] = []
    slices = evidence["slices"]
    manifest_scenarios = evidence["manifest"].get("scenarios") if isinstance(evidence["manifest"].get("scenarios"), list) else []
    slice_ids = {str(row.get("scenario_id")) for row in slices if row.get("scenario_id")}
    manifest_ids = {str(row.get("scenario_id")) for row in manifest_scenarios if isinstance(row, dict) and row.get("scenario_id")}
    scenario_id_mismatch_count = len(slice_ids.symmetric_difference(manifest_ids))
    if scenario_id_mismatch_count:
        reasons.append("manifest_slice_scenario_id_drift")
    missing_contract_paths: list[str] = []
    missing_sidecar_paths: list[str] = []
    for row in slices:
        contract = _resolve_source_file(row.get("contract"), repo_root)
        sidecar = _resolve_source_file(row.get("sidecar"), repo_root)
        if not contract or not contract.is_file():
            missing_contract_paths.append(str(row.get("contract")))
        if not sidecar or not sidecar.is_file():
            missing_sidecar_paths.append(str(row.get("sidecar")))
    if config["require_all_contract_and_sidecar_paths"] and (missing_contract_paths or missing_sidecar_paths):
        reasons.append("real_map_path_feedback_contract_incomplete")
    return {
        "schema_version": "global-99-real-map-manifest-sidecar-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "slice_count": len(slices),
        "manifest_scenario_count": len(manifest_scenarios),
        "scenario_id_mismatch_count": scenario_id_mismatch_count,
        "missing_contract_count": len(missing_contract_paths),
        "missing_sidecar_count": len(missing_sidecar_paths),
        "missing_contract_paths": missing_contract_paths,
        "missing_sidecar_paths": missing_sidecar_paths,
    }


def _context_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    domain = evidence["domain_gap"]
    bridge = evidence["bridge"]
    slices = evidence["slices"]
    context_missing = max(_int_value(domain.get("context_id_missing_count")), _int_value(bridge.get("context_id_missing_count")))
    legacy = max(_int_value(domain.get("legacy_identity_fallback_count")), _int_value(bridge.get("legacy_identity_fallback_count")))
    context_missing = max(context_missing, sum(1 for row in slices if not row.get("context_id")))
    legacy = max(legacy, sum(1 for row in slices if row.get("legacy_identity_fallback_used") is True))
    reasons: list[str] = []
    if (config["require_context_ids"] and context_missing) or (config["require_no_legacy_identity_fallback"] and legacy):
        reasons.append("real_map_context_identity_incomplete")
    return {
        "schema_version": "global-99-real-map-context-audit/v1",
        "passed": not reasons,
        "reason_codes": reasons,
        "context_id_missing_count": context_missing,
        "legacy_identity_fallback_count": legacy,
    }


def _source_match_drift_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    shadow_canary = evidence["shadow_canary"]
    shadow_replay = evidence["shadow_replay"]
    reasons: list[str] = []
    source_match = bool(shadow_canary.get("source_match_audit_passed", False) and shadow_replay.get("source_match_audit_passed", True))
    scenario_mismatch = max(_int_value(shadow_canary.get("scenario_mismatch_count")), _int_value(shadow_replay.get("scenario_mismatch_count")))
    coverage_delta = max(_float_default(shadow_canary.get("max_replay_coverage_delta")), _float_default(shadow_replay.get("max_replay_coverage_delta")))
    path_delta = max(_float_default(shadow_canary.get("max_replay_path_cost_delta_m")), _float_default(shadow_replay.get("max_replay_path_cost_delta_m")))
    fallback_rate = _float_default(shadow_canary.get("policy_guard_fallback_rate"))
    if config["require_source_match_audit_passed"] and (not source_match or scenario_mismatch):
        reasons.append("real_map_source_match_drift_detected")
    if coverage_delta > config["source_match_coverage_tolerance"] or path_delta > config["source_match_path_cost_tolerance_m"]:
        reasons.append("real_map_source_match_drift_detected")
    if fallback_rate > config["max_policy_guard_fallback_rate"]:
        reasons.append("policy_guard_fallback_rate_too_high")
    return {
        "schema_version": "global-99-real-map-source-match-drift-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "source_match_audit_passed": source_match,
        "scenario_mismatch_count": scenario_mismatch,
        "max_replay_coverage_delta": coverage_delta,
        "max_replay_path_cost_delta_m": path_delta,
        "policy_guard_fallback_rate": fallback_rate,
    }


def _fingerprint_audit(source_paths: dict[str, Path], manifest_sidecar: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for label, path in sorted(source_paths.items()):
        if label.endswith("_root"):
            continue
        if path.is_file():
            records.append({"label": label, "path": str(path), "sha256": _sha256(path)})
        else:
            missing.append(str(path))
    for path_string in manifest_sidecar.get("missing_contract_paths", []) + manifest_sidecar.get("missing_sidecar_paths", []):
        if path_string:
            missing.append(path_string)
    reasons = ["real_map_evidence_fingerprint_incomplete"] if missing else []
    return {
        "schema_version": "global-99-real-map-evidence-fingerprint-audit/v1",
        "passed": not reasons,
        "reason_codes": reasons,
        "fingerprint_count": len(records),
        "missing_fingerprint_source_count": len(missing),
        "missing_sources": sorted(set(missing)),
        "fingerprints": records,
    }


def _boundary_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for source_name in ("shadow_canary", "shadow_replay", "domain_gap", "path_feedback"):
        source = evidence[source_name]
        for field in FORBIDDEN_FIELDS:
            if (
                source_name == "shadow_replay"
                and field == "uses_path_planner"
                and source.get(field) is True
                and source.get("path_planner_use_scope") == "offline_path_feedback_replay_only"
            ):
                continue
            if source.get(field) is True:
                violations.append({"source": source_name, "field": field, "value": True})
        if _float_default(source.get("canary_traffic_fraction")) > 0:
            violations.append({"source": source_name, "field": "canary_traffic_fraction", "value": source.get("canary_traffic_fraction")})
    if config["canary_traffic_fraction"] > 0:
        violations.append({"source": "config", "field": "canary_traffic_fraction", "value": config["canary_traffic_fraction"]})
    reasons = ["real_map_evidence_refresh_boundary_violation"] if violations else []
    return {
        "schema_version": "global-99-real-map-evidence-refresh-drift-boundary-audit/v1",
        "passed": not reasons,
        "reason_codes": reasons,
        "violations": violations,
        **BOUNDARY_FIELDS,
    }


def _decision(
    lineage: dict[str, Any],
    fingerprint: dict[str, Any],
    manifest_sidecar: dict[str, Any],
    context: dict[str, Any],
    source_match: dict[str, Any],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    reasons = unique_sorted(
        list(lineage["reason_codes"])
        + list(fingerprint["reason_codes"])
        + list(manifest_sidecar["reason_codes"])
        + list(context["reason_codes"])
        + list(source_match["reason_codes"])
        + list(boundary["reason_codes"])
    )
    status = "passed" if not reasons else "failed"
    next_required_change = PASS_NEXT_REQUIRED_CHANGE
    if status == "failed":
        if any(reason.startswith("missing_real_map_shadow_canary_replay") or reason.startswith("real_map_shadow_canary_replay") for reason in reasons):
            next_required_change = FIX_SHADOW_CANARY_NEXT_REQUIRED_CHANGE
        elif "domain_gap_evidence_not_acceptable" in reasons or any(reason.startswith("missing_domain_gap") for reason in reasons):
            next_required_change = FIX_DOMAIN_GAP_NEXT_REQUIRED_CHANGE
        elif "real_map_context_identity_incomplete" in reasons:
            next_required_change = FIX_CONTEXT_NEXT_REQUIRED_CHANGE
        elif "real_map_path_feedback_contract_incomplete" in reasons:
            next_required_change = FIX_PATH_FEEDBACK_NEXT_REQUIRED_CHANGE
        elif "manifest_slice_scenario_id_drift" in reasons or "real_map_evidence_fingerprint_incomplete" in reasons:
            next_required_change = FIX_DRIFT_NEXT_REQUIRED_CHANGE
        elif "real_map_source_match_drift_detected" in reasons:
            next_required_change = FIX_DETERMINISM_NEXT_REQUIRED_CHANGE
        elif "policy_guard_fallback_rate_too_high" in reasons:
            next_required_change = FIX_FALLBACK_NEXT_REQUIRED_CHANGE
        elif "real_map_evidence_refresh_boundary_violation" in reasons:
            next_required_change = BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE
        else:
            next_required_change = FIX_DRIFT_NEXT_REQUIRED_CHANGE
    return {
        "status": status,
        "reason_codes": reasons,
        "verdict": PASS_VERDICT if status == "passed" else "blocked",
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
    fingerprint: dict[str, Any],
    manifest_sidecar: dict[str, Any],
    context: dict[str, Any],
    source_match: dict[str, Any],
    boundary: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    domain = evidence["domain_gap"]
    path_feedback = evidence["path_feedback"]
    regression_count = max(
        _int_value(domain.get("fallback_or_open_grid_count")),
        _int_value(domain.get("open_grid_fallback_count")),
        int(bool(path_feedback.get("open_grid_fallback_used"))),
        _int_value(domain.get("safety_regression_count")),
        _int_value(domain.get("contract_violation_count")),
        _int_value(domain.get("path_cost_regression_count")),
        _int_value(domain.get("risk_regression_count")),
        _int_value(domain.get("source_selection_regression_count")),
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_real_map_shadow_canary_replay_status": lineage["source_real_map_shadow_canary_replay_status"],
        "source_real_map_shadow_canary_replay_next_required_change": lineage["source_real_map_shadow_canary_replay_next_required_change"],
        "source_domain_gap_status": lineage["source_domain_gap_status"],
        "domain_gap_verdict": lineage["domain_gap_verdict"],
        "source_path_feedback_summary_present": bool(path_feedback),
        "slice_count": lineage["slice_count"],
        "roi_group_count": lineage["roi_group_count"],
        "manifest_scenario_count": manifest_sidecar["manifest_scenario_count"],
        "scenario_id_mismatch_count": manifest_sidecar["scenario_id_mismatch_count"],
        "missing_contract_count": manifest_sidecar["missing_contract_count"],
        "missing_sidecar_count": manifest_sidecar["missing_sidecar_count"],
        "context_id_missing_count": context["context_id_missing_count"],
        "legacy_identity_fallback_count": context["legacy_identity_fallback_count"],
        "fallback_or_open_grid_count": _int_value(domain.get("fallback_or_open_grid_count")) or int(bool(path_feedback.get("open_grid_fallback_used"))),
        "safety_regression_count": _int_value(domain.get("safety_regression_count")),
        "contract_violation_count": _int_value(domain.get("contract_violation_count")),
        "path_cost_regression_count": _int_value(domain.get("path_cost_regression_count")),
        "risk_regression_count": _int_value(domain.get("risk_regression_count")),
        "source_selection_regression_count": _int_value(domain.get("source_selection_regression_count")),
        "source_match_audit_passed": source_match["source_match_audit_passed"],
        "scenario_mismatch_count": source_match["scenario_mismatch_count"],
        "max_replay_coverage_delta": source_match["max_replay_coverage_delta"],
        "max_replay_path_cost_delta_m": source_match["max_replay_path_cost_delta_m"],
        "policy_guard_fallback_rate": source_match["policy_guard_fallback_rate"],
        "evidence_fingerprint_count": fingerprint["fingerprint_count"],
        "missing_fingerprint_source_count": fingerprint["missing_fingerprint_source_count"],
        "lineage_audit_passed": lineage["passed"],
        "fingerprint_audit_passed": fingerprint["passed"],
        "manifest_sidecar_audit_passed": manifest_sidecar["passed"],
        "context_audit_passed": context["passed"],
        "source_match_drift_audit_passed": source_match["passed"],
        "boundary_audit_passed": boundary["passed"],
        "kill_switch_audit_passed": kill_switch["passed"],
        "rollback_audit_passed": rollback["passed"],
        "telemetry_audit_passed": telemetry["passed"],
        "evidence_refresh_drift_audit_passed": decision["status"] == "passed",
        "evidence_refresh_drift_verdict": decision["verdict"],
        "path_feedback_regression_or_fallback_count": regression_count,
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


def _simple_audit(name: str, passed: bool) -> dict[str, Any]:
    return {
        "schema_version": f"global-99-real-map-evidence-refresh-drift-{name}-audit/v1",
        "passed": bool(passed),
        "reason_codes": [] if passed else [f"{name}_audit_blocked"],
    }


def _render_report(summary: dict[str, Any], rejection: dict[str, Any]) -> str:
    return "\n".join([
        "# Global 99 Real Map Evidence Refresh / Drift Audit v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- verdict: `{summary['evidence_refresh_drift_verdict']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        "",
        "## Rejections",
        "",
        f"```json\n{json.dumps(rejection, ensure_ascii=False, indent=2)}\n```",
        "",
    ])


def _resolve_source_file(value: Any, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else resolve_path(path, repo_root)


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
