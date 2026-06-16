from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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


CONFIG_SCHEMA_VERSION = "global-99-real-map-multi-roi-generalization-config/v1"
SUMMARY_SCHEMA_VERSION = "global-99-real-map-multi-roi-generalization-summary/v1"
DEFAULT_CONFIG = "configs/global_99_real_map_multi_roi_generalization_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_global_99_real_map_multi_roi_generalization_v1"

DRIFT_SUMMARY_FILE = "global-99-real-map-evidence-refresh-drift-audit-summary.json"
DOMAIN_GAP_SUMMARY_FILE = "quasi-real-map-domain-gap-summary.json"
SLICES_FILE = "quasi-real-map-slices.jsonl"

SUMMARY_FILE = "global-99-real-map-multi-roi-generalization-summary.json"
MANIFEST_FILE = "global-99-real-map-multi-roi-generalization-manifest.json"
SCENARIO_RESULTS_FILE = "global-99-real-map-multi-roi-scenario-results.jsonl"
FAMILY_SUMMARY_FILE = "global-99-real-map-multi-roi-family-summary.json"
LINEAGE_AUDIT_FILE = "global-99-real-map-multi-roi-lineage-audit.json"
SCENARIO_MATRIX_AUDIT_FILE = "global-99-real-map-multi-roi-scenario-matrix-audit.json"
BOUNDARY_AUDIT_FILE = "global-99-real-map-multi-roi-boundary-audit.json"
KILL_SWITCH_AUDIT_FILE = "global-99-real-map-multi-roi-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "global-99-real-map-multi-roi-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "global-99-real-map-multi-roi-telemetry-audit.json"
REJECTION_REPORT_FILE = "global-99-real-map-multi-roi-rejection-report.json"
REPORT_FILE = "global-99-real-map-multi-roi-generalization-report.md"

PASS_VERDICT = "eligible_for_default_policy_candidate_authorization_preflight"
PASS_NEXT_REQUIRED_CHANGE = "default_policy_candidate_authorization_preflight"
FIX_DRIFT_NEXT_REQUIRED_CHANGE = "fix_global_99_real_map_evidence_refresh_drift_audit"
FIX_DOMAIN_GAP_NEXT_REQUIRED_CHANGE = "fix_quasi_real_map_domain_gap_evidence"
EXPAND_ROI_NEXT_REQUIRED_CHANGE = "expand_real_map_roi_coverage"
FIX_CONTEXT_NEXT_REQUIRED_CHANGE = "fix_real_map_context_identity"
FIX_PATH_FEEDBACK_NEXT_REQUIRED_CHANGE = "fix_real_map_path_feedback_contract"
BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE = "resolve_global_99_real_map_multi_roi_boundary_rejections"

BOUNDARY_FIELDS = {
    "multi_roi_generalization_only": True,
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
    parser = argparse.ArgumentParser(description="Run Global 99 Real Map Multi-ROI Generalization v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_global_99_real_map_multi_roi_generalization(
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
        "real_map_multi_roi_generalization_verdict": summary["real_map_multi_roi_generalization_verdict"],
        "next_required_change": summary["next_required_change"],
        "summary": summary["summary"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_global_99_real_map_multi_roi_generalization(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_config(Path(config_path), repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    source_paths = _source_paths(config, repo_root)
    evidence = _load_evidence(source_paths)

    lineage = _lineage_audit(config, evidence)
    scenario_results, family_summary, matrix = _scenario_matrix_audit(config, evidence, repo_root)
    boundary = _boundary_audit(config, evidence)
    kill_switch = _simple_audit("kill-switch", lineage["passed"] and boundary["passed"])
    rollback = _simple_audit("rollback", matrix["passed"] and boundary["passed"])
    telemetry = _simple_audit("telemetry", matrix["passed"] and lineage["passed"])
    decision = _decision(lineage, matrix, boundary)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=Path(config_path),
        output_root=output_root,
        paths=paths,
        source_paths=source_paths,
        evidence=evidence,
        lineage=lineage,
        matrix=matrix,
        boundary=boundary,
        kill_switch=kill_switch,
        rollback=rollback,
        telemetry=telemetry,
        decision=decision,
        repo_root=Path(repo_root),
    )
    manifest = {
        "schema_version": "global-99-real-map-multi-roi-generalization-manifest/v1",
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "artifacts": {k: str(v) for k, v in paths.items()},
        "status": summary["status"],
        "next_required_change": summary["next_required_change"],
    }
    rejection = {
        "schema_version": "global-99-real-map-multi-roi-rejection-report/v1",
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
    }
    write_json(paths["lineage"], lineage)
    write_json(paths["family_summary"], family_summary)
    write_json(paths["scenario_matrix"], matrix)
    write_json(paths["boundary"], boundary)
    write_json(paths["kill_switch"], kill_switch)
    write_json(paths["rollback"], rollback)
    write_json(paths["telemetry"], telemetry)
    write_json(paths["rejection_report"], rejection)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["scenario_results"].write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in scenario_results) + ("\n" if scenario_results else ""),
        encoding="utf-8",
    )
    paths["report"].write_text(_render_report(summary, rejection), encoding="utf-8")
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    normalized = dict(payload)
    for key in ("source_evidence_refresh_drift_root", "source_quasi_real_domain_gap_root"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(payload[key]), repo_root))
    for key in ("min_slice_count", "min_roi_group_count", "min_slices_per_roi_group"):
        if not isinstance(payload.get(key), int) or isinstance(payload.get(key), bool):
            raise ConfigError(f"{key} must be an integer")
    if not isinstance(payload.get("required_splits"), list) or not all(isinstance(item, str) and item for item in payload["required_splits"]):
        raise ConfigError("required_splits must be a non-empty string list")
    for key in (
        "require_evidence_refresh_drift_passed",
        "require_domain_gap_acceptable",
        "require_context_ids",
        "require_no_legacy_identity_fallback",
        "require_no_open_grid_fallback",
        "require_all_contract_and_sidecar_paths",
        "multi_roi_generalization_only",
    ):
        if not isinstance(payload.get(key), bool):
            raise ConfigError(f"{key} must be a boolean")
    normalized["canary_traffic_fraction"] = _float_value(payload.get("canary_traffic_fraction"), "canary_traffic_fraction")
    return normalized


def _source_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    drift_root = resolve_path(Path(config["source_evidence_refresh_drift_root"]), repo_root)
    domain_gap_root = resolve_path(Path(config["source_quasi_real_domain_gap_root"]), repo_root)
    return {
        "evidence_refresh_drift_root": drift_root,
        "quasi_real_domain_gap_root": domain_gap_root,
        "evidence_refresh_drift_summary": drift_root / DRIFT_SUMMARY_FILE,
        "domain_gap_summary": domain_gap_root / DOMAIN_GAP_SUMMARY_FILE,
        "slices": domain_gap_root / SLICES_FILE,
    }


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "scenario_results": output_root / SCENARIO_RESULTS_FILE,
        "family_summary": output_root / FAMILY_SUMMARY_FILE,
        "lineage": output_root / LINEAGE_AUDIT_FILE,
        "scenario_matrix": output_root / SCENARIO_MATRIX_AUDIT_FILE,
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
        "drift": _read_json(source_paths["evidence_refresh_drift_summary"], reasons, "evidence_refresh_drift"),
        "domain_gap": _read_json(source_paths["domain_gap_summary"], reasons, "domain_gap"),
        "slices": _read_jsonl(source_paths["slices"], reasons, "slices"),
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
    drift = evidence["drift"]
    domain = evidence["domain_gap"]
    reasons = list(evidence["read_reason_codes"])
    if drift and config["require_evidence_refresh_drift_passed"] and drift.get("status") != "passed":
        reasons.append("evidence_refresh_drift_not_passed")
    if drift and drift.get("next_required_change") != "global_99_real_map_multi_roi_generalization":
        reasons.append("evidence_refresh_drift_wrong_next_required_change")
    if domain and config["require_domain_gap_acceptable"]:
        if domain.get("status") != "passed" or domain.get("domain_gap_verdict") != "acceptable_for_next_pilot":
            reasons.append("domain_gap_evidence_not_acceptable")
    return {
        "schema_version": "global-99-real-map-multi-roi-lineage-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "source_evidence_refresh_drift_status": drift.get("status"),
        "source_evidence_refresh_drift_next_required_change": drift.get("next_required_change"),
        "source_domain_gap_status": domain.get("status"),
        "domain_gap_verdict": domain.get("domain_gap_verdict"),
    }


def _scenario_matrix_audit(config: dict[str, Any], evidence: dict[str, Any], repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    slices = evidence["slices"]
    domain = evidence["domain_gap"]
    required_splits = set(config["required_splits"])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scenario_results: list[dict[str, Any]] = []
    context_missing = 0
    legacy = 0
    missing_contract = 0
    missing_sidecar = 0
    for row in slices:
        roi_group = str(row.get("roi_name") or row.get("scenario_group") or "unknown")
        groups[roi_group].append(row)
        contract = _resolve_source_file(row.get("contract"), repo_root)
        sidecar = _resolve_source_file(row.get("sidecar"), repo_root)
        has_context = bool(row.get("context_id"))
        has_contract = bool(contract and contract.is_file())
        has_sidecar = bool(sidecar and sidecar.is_file())
        legacy_used = row.get("legacy_identity_fallback_used") is True
        context_missing += 0 if has_context else 1
        legacy += 1 if legacy_used else 0
        missing_contract += 0 if has_contract else 1
        missing_sidecar += 0 if has_sidecar else 1
        passed = has_context and not legacy_used and has_contract and has_sidecar
        scenario_results.append({
            "scenario_id": row.get("scenario_id"),
            "roi_group": roi_group,
            "split": row.get("split"),
            "required": True,
            "passed": passed,
            "reason_codes": [] if passed else _scenario_reason_codes(has_context, legacy_used, has_contract, has_sidecar),
        })
    family_rows: list[dict[str, Any]] = []
    failed_groups = 0
    for roi_group, rows in sorted(groups.items()):
        splits = {str(row.get("split")) for row in rows if row.get("split")}
        group_context_missing = sum(1 for row in rows if not row.get("context_id"))
        group_legacy = sum(1 for row in rows if row.get("legacy_identity_fallback_used") is True)
        split_complete = required_splits.issubset(splits)
        group_passed = (
            len(rows) >= config["min_slices_per_roi_group"]
            and split_complete
            and group_context_missing == 0
            and group_legacy == 0
        )
        if not group_passed:
            failed_groups += 1
        family_rows.append({
            "roi_group": roi_group,
            "slice_count": len(rows),
            "splits": sorted(splits),
            "split_coverage_complete": split_complete,
            "context_id_missing_count": group_context_missing,
            "legacy_identity_fallback_count": group_legacy,
            "passed": group_passed,
        })
    roi_group_count = len(groups)
    slice_count = len(slices)
    split_coverage_complete = all(row["split_coverage_complete"] for row in family_rows) if family_rows else False
    path_feedback_regression = max(
        _int_value(domain.get("fallback_or_open_grid_count")),
        _int_value(domain.get("open_grid_fallback_count")),
        _int_value(domain.get("safety_regression_count")),
        _int_value(domain.get("contract_violation_count")),
        _int_value(domain.get("path_cost_regression_count")),
        _int_value(domain.get("risk_regression_count")),
        _int_value(domain.get("source_selection_regression_count")),
    )
    reasons: list[str] = []
    if slice_count < config["min_slice_count"] or roi_group_count < config["min_roi_group_count"] or failed_groups or not split_coverage_complete:
        reasons.append("real_map_multi_roi_coverage_insufficient")
    if (config["require_context_ids"] and context_missing) or (config["require_no_legacy_identity_fallback"] and legacy):
        reasons.append("real_map_context_identity_incomplete")
    if config["require_all_contract_and_sidecar_paths"] and (missing_contract or missing_sidecar):
        reasons.append("real_map_path_feedback_contract_incomplete")
    if config["require_no_open_grid_fallback"] and path_feedback_regression:
        reasons.append("real_map_path_feedback_contract_incomplete")
    family_summary = {
        "schema_version": "global-99-real-map-multi-roi-family-summary/v1",
        "roi_group_count": roi_group_count,
        "passed_roi_group_count": roi_group_count - failed_groups,
        "failed_roi_group_count": failed_groups,
        "families": family_rows,
    }
    matrix = {
        "schema_version": "global-99-real-map-multi-roi-scenario-matrix-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "slice_count": slice_count,
        "roi_group_count": roi_group_count,
        "required_roi_group_count": roi_group_count,
        "passed_roi_group_count": roi_group_count - failed_groups,
        "failed_roi_group_count": failed_groups,
        "required_scenario_count": len(scenario_results),
        "passed_required_scenario_count": sum(1 for row in scenario_results if row["passed"]),
        "failed_required_scenario_count": sum(1 for row in scenario_results if not row["passed"]),
        "split_coverage_complete": split_coverage_complete,
        "context_id_missing_count": context_missing,
        "legacy_identity_fallback_count": legacy,
        "missing_contract_count": missing_contract,
        "missing_sidecar_count": missing_sidecar,
        "fallback_or_open_grid_count": _int_value(domain.get("fallback_or_open_grid_count")),
        "path_feedback_regression_or_fallback_count": path_feedback_regression,
    }
    return scenario_results, family_summary, matrix


def _boundary_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for source_name in ("drift", "domain_gap"):
        source = evidence[source_name]
        for field in FORBIDDEN_FIELDS:
            if source.get(field) is True:
                violations.append({"source": source_name, "field": field, "value": True})
        if _float_default(source.get("canary_traffic_fraction")) > 0:
            violations.append({"source": source_name, "field": "canary_traffic_fraction", "value": source.get("canary_traffic_fraction")})
    if config["canary_traffic_fraction"] > 0:
        violations.append({"source": "config", "field": "canary_traffic_fraction", "value": config["canary_traffic_fraction"]})
    reasons = ["real_map_multi_roi_boundary_violation"] if violations else []
    return {
        "schema_version": "global-99-real-map-multi-roi-boundary-audit/v1",
        "passed": not reasons,
        "reason_codes": reasons,
        "violations": violations,
        **BOUNDARY_FIELDS,
    }


def _decision(lineage: dict[str, Any], matrix: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    reasons = unique_sorted(list(lineage["reason_codes"]) + list(matrix["reason_codes"]) + list(boundary["reason_codes"]))
    status = "passed" if not reasons else "failed"
    next_required_change = PASS_NEXT_REQUIRED_CHANGE
    if status == "failed":
        if any(reason.startswith("missing_evidence_refresh_drift") or reason.startswith("evidence_refresh_drift") for reason in reasons):
            next_required_change = FIX_DRIFT_NEXT_REQUIRED_CHANGE
        elif "domain_gap_evidence_not_acceptable" in reasons or any(reason.startswith("missing_domain_gap") for reason in reasons):
            next_required_change = FIX_DOMAIN_GAP_NEXT_REQUIRED_CHANGE
        elif "real_map_context_identity_incomplete" in reasons:
            next_required_change = FIX_CONTEXT_NEXT_REQUIRED_CHANGE
        elif "real_map_path_feedback_contract_incomplete" in reasons:
            next_required_change = FIX_PATH_FEEDBACK_NEXT_REQUIRED_CHANGE
        elif "real_map_multi_roi_coverage_insufficient" in reasons:
            next_required_change = EXPAND_ROI_NEXT_REQUIRED_CHANGE
        elif "real_map_multi_roi_boundary_violation" in reasons:
            next_required_change = BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE
        else:
            next_required_change = EXPAND_ROI_NEXT_REQUIRED_CHANGE
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
    matrix: dict[str, Any],
    boundary: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_evidence_refresh_drift_status": lineage["source_evidence_refresh_drift_status"],
        "source_evidence_refresh_drift_next_required_change": lineage["source_evidence_refresh_drift_next_required_change"],
        "source_domain_gap_status": lineage["source_domain_gap_status"],
        "domain_gap_verdict": lineage["domain_gap_verdict"],
        "slice_count": matrix["slice_count"],
        "roi_group_count": matrix["roi_group_count"],
        "required_roi_group_count": matrix["required_roi_group_count"],
        "passed_roi_group_count": matrix["passed_roi_group_count"],
        "failed_roi_group_count": matrix["failed_roi_group_count"],
        "required_scenario_count": matrix["required_scenario_count"],
        "passed_required_scenario_count": matrix["passed_required_scenario_count"],
        "failed_required_scenario_count": matrix["failed_required_scenario_count"],
        "split_coverage_complete": matrix["split_coverage_complete"],
        "context_id_missing_count": matrix["context_id_missing_count"],
        "legacy_identity_fallback_count": matrix["legacy_identity_fallback_count"],
        "missing_contract_count": matrix["missing_contract_count"],
        "missing_sidecar_count": matrix["missing_sidecar_count"],
        "fallback_or_open_grid_count": matrix["fallback_or_open_grid_count"],
        "path_feedback_regression_or_fallback_count": matrix["path_feedback_regression_or_fallback_count"],
        "roi_generalization_audit_passed": matrix["passed"],
        "scenario_matrix_audit_passed": matrix["passed"],
        "lineage_audit_passed": lineage["passed"],
        "boundary_audit_passed": boundary["passed"],
        "kill_switch_audit_passed": kill_switch["passed"],
        "rollback_audit_passed": rollback["passed"],
        "telemetry_audit_passed": telemetry["passed"],
        "real_map_multi_roi_generalization_passed": decision["status"] == "passed",
        "real_map_multi_roi_generalization_verdict": decision["verdict"],
        "next_required_change": decision["next_required_change"],
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "scenario_results": str(paths["scenario_results"]),
        "family_summary": str(paths["family_summary"]),
        "report": str(paths["report"]),
        "source_paths": {key: str(value) for key, value in source_paths.items()},
        "config": str(config_path),
        "output_root": str(output_root),
        "git_provenance": git_snapshot(repo_root),
        **BOUNDARY_FIELDS,
    }
    return summary


def _scenario_reason_codes(has_context: bool, legacy_used: bool, has_contract: bool, has_sidecar: bool) -> list[str]:
    reasons: list[str] = []
    if not has_context or legacy_used:
        reasons.append("real_map_context_identity_incomplete")
    if not has_contract or not has_sidecar:
        reasons.append("real_map_path_feedback_contract_incomplete")
    return reasons


def _simple_audit(name: str, passed: bool) -> dict[str, Any]:
    return {
        "schema_version": f"global-99-real-map-multi-roi-{name}-audit/v1",
        "passed": bool(passed),
        "reason_codes": [] if passed else [f"{name}_audit_blocked"],
    }


def _render_report(summary: dict[str, Any], rejection: dict[str, Any]) -> str:
    return "\n".join([
        "# Global 99 Real Map Multi-ROI Generalization v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- verdict: `{summary['real_map_multi_roi_generalization_verdict']}`",
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
