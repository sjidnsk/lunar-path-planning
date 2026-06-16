from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl


CONFIG_SCHEMA_VERSION = "global-99-real-map-shadow-replay-config/v1"
SUMMARY_SCHEMA_VERSION = "global-99-real-map-shadow-replay-summary/v1"
MANIFEST_SCHEMA_VERSION = "global-99-real-map-shadow-replay-manifest/v1"
SOURCE_MATCH_AUDIT_SCHEMA_VERSION = "global-99-real-map-shadow-replay-source-match-audit/v1"
CONTEXT_AUDIT_SCHEMA_VERSION = "global-99-real-map-shadow-replay-context-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "global-99-real-map-shadow-replay-boundary-audit/v1"
KILL_SWITCH_AUDIT_SCHEMA_VERSION = "global-99-real-map-shadow-replay-kill-switch-audit/v1"
ROLLBACK_AUDIT_SCHEMA_VERSION = "global-99-real-map-shadow-replay-rollback-audit/v1"
TELEMETRY_AUDIT_SCHEMA_VERSION = "global-99-real-map-shadow-replay-telemetry-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "global-99-real-map-shadow-replay-rejection-report/v1"

DEFAULT_CONFIG = "configs/global_99_real_map_shadow_replay_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_global_99_real_map_shadow_replay_v1"

PREFLIGHT_SUMMARY_FILE = "global-99-real-map-preflight-summary.json"
DOMAIN_GAP_SUMMARY_FILE = "quasi-real-map-domain-gap-summary.json"
SOURCE_PATH_FEEDBACK_MANIFEST_FILE = "quasi-real-map-path-feedback-manifest.json"
SOURCE_PATH_FEEDBACK_SUMMARY_FILE = "quasi-real-map-path-feedback-summary.json"
SOURCE_SLICES_FILE = "quasi-real-map-slices.jsonl"

SUMMARY_FILE = "global-99-real-map-shadow-replay-summary.json"
MANIFEST_FILE = "global-99-real-map-shadow-replay-manifest.json"
REPLAY_PATH_FEEDBACK_MANIFEST_FILE = "global-99-real-map-shadow-replay-path-feedback-manifest.json"
REPLAY_PATH_FEEDBACK_SUMMARY_FILE = "global-99-real-map-shadow-replay-path-feedback-summary.json"
REPLAY_PATH_FEEDBACK_REPORT_FILE = "global-99-real-map-shadow-replay-path-feedback-report.md"
SCENARIO_RESULTS_FILE = "global-99-real-map-shadow-replay-scenario-results.jsonl"
SOURCE_MATCH_AUDIT_FILE = "global-99-real-map-shadow-replay-source-match-audit.json"
CONTEXT_AUDIT_FILE = "global-99-real-map-shadow-replay-context-audit.json"
BOUNDARY_AUDIT_FILE = "global-99-real-map-shadow-replay-boundary-audit.json"
KILL_SWITCH_AUDIT_FILE = "global-99-real-map-shadow-replay-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "global-99-real-map-shadow-replay-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "global-99-real-map-shadow-replay-telemetry-audit.json"
REJECTION_REPORT_FILE = "global-99-real-map-shadow-replay-rejection-report.json"
REPORT_FILE = "global-99-real-map-shadow-replay-report.md"

PASS_VERDICT = "eligible_for_global_99_real_map_release_governance_preflight"
PASS_NEXT_REQUIRED_CHANGE = "global_99_real_map_release_governance_preflight"
FIX_PREFLIGHT_NEXT_REQUIRED_CHANGE = "fix_global_99_real_map_preflight"
FIX_DOMAIN_GAP_NEXT_REQUIRED_CHANGE = "fix_quasi_real_map_domain_gap_evidence"
FIX_REPLAY_NEXT_REQUIRED_CHANGE = "fix_global_99_real_map_shadow_replay"
FIX_DETERMINISM_NEXT_REQUIRED_CHANGE = "fix_global_99_real_map_shadow_replay_determinism"
FIX_CONTEXT_NEXT_REQUIRED_CHANGE = "fix_real_map_context_identity"
FIX_PATH_FEEDBACK_NEXT_REQUIRED_CHANGE = "fix_real_map_path_feedback_contract"
BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE = "resolve_global_99_real_map_shadow_replay_boundary_rejections"

BOUNDARY_FIELDS = {
    "uses_lola_quasi_real_roi": True,
    "uses_path_feedback_sidecar": True,
    "uses_offline_path_feedback_replay": True,
    "uses_path_planner": True,
    "path_planner_use_scope": "offline_path_feedback_replay_only",
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
    parser = argparse.ArgumentParser(description="Run Global 99 Real Map Shadow Replay v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_global_99_real_map_shadow_replay(
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
                "real_map_shadow_replay_verdict": summary["real_map_shadow_replay_verdict"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_global_99_real_map_shadow_replay(
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
    replay_manifest = _build_replay_manifest(evidence["source_path_feedback_manifest"], paths)

    lineage = _lineage_audit(config, evidence)
    context = _context_audit(config, evidence)
    source_contract = _source_contract_audit(config, evidence)
    boundary = _boundary_audit(config, evidence)
    pre_replay_reason_codes = unique_sorted(
        list(lineage["reason_codes"])
        + list(context["reason_codes"])
        + list(source_contract["reason_codes"])
        + list(boundary["reason_codes"])
    )
    replay_run = _skip_replay(pre_replay_reason_codes)
    if not pre_replay_reason_codes and replay_manifest:
        write_json(paths["replay_path_feedback_manifest"], replay_manifest)
        replay_run = _run_path_feedback_replay(paths["replay_path_feedback_manifest"], repo_root)
    elif replay_manifest:
        write_json(paths["replay_path_feedback_manifest"], replay_manifest)

    replay_summary = _read_json(paths["replay_path_feedback_summary"], [], "replay_path_feedback_summary")
    source_match = _source_match_audit(config, evidence, replay_summary, replay_run)
    scenario_results = source_match["scenario_results"]
    kill_switch = _kill_switch_audit(source_match, boundary)
    rollback = _rollback_audit(source_match, boundary)
    telemetry = _telemetry_audit(evidence, replay_summary, replay_run, source_match, context)
    decision = _decision(
        evidence,
        lineage,
        context,
        source_contract,
        source_match,
        replay_run,
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
        context=context,
        source_match=source_match,
        replay_run=replay_run,
        boundary=boundary,
        kill_switch=kill_switch,
        rollback=rollback,
        telemetry=telemetry,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)
    rejection_report = _rejection_report(generated_at, decision, lineage, context, source_contract, source_match, replay_run, boundary)

    write_jsonl(paths["scenario_results"], scenario_results)
    for key, payload in (
        ("source_match", source_match),
        ("context", context),
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
    for key in ("source_real_map_preflight_root", "source_quasi_real_domain_gap_root"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    normalized = dict(payload)
    normalized["source_real_map_preflight_root"] = str(resolve_path(Path(payload["source_real_map_preflight_root"]), repo_root))
    normalized["source_quasi_real_domain_gap_root"] = str(resolve_path(Path(payload["source_quasi_real_domain_gap_root"]), repo_root))
    normalized["target_coverage_rate"] = _bounded_float(payload.get("target_coverage_rate"), "target_coverage_rate")
    normalized["source_match_coverage_tolerance"] = _non_negative_float(
        payload.get("source_match_coverage_tolerance"),
        "source_match_coverage_tolerance",
    )
    normalized["source_match_path_cost_tolerance_m"] = _non_negative_float(
        payload.get("source_match_path_cost_tolerance_m"),
        "source_match_path_cost_tolerance_m",
    )
    for key in (
        "require_preflight_passed",
        "require_context_ids",
        "require_no_open_grid_fallback",
        "offline_path_feedback_replay",
        "allow_offline_path_planner_route_replay",
        "shadow_replay_mode",
    ):
        normalized[key] = _bool_value(payload.get(key), key)
    normalized["canary_traffic_fraction"] = _fraction_float(payload.get("canary_traffic_fraction"), "canary_traffic_fraction")
    return normalized


def _source_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    preflight_root = resolve_path(Path(config["source_real_map_preflight_root"]), repo_root)
    domain_gap_root = resolve_path(Path(config["source_quasi_real_domain_gap_root"]), repo_root)
    return {
        "real_map_preflight_root": preflight_root,
        "quasi_real_domain_gap_root": domain_gap_root,
        "real_map_preflight_summary": preflight_root / PREFLIGHT_SUMMARY_FILE,
        "domain_gap_summary": domain_gap_root / DOMAIN_GAP_SUMMARY_FILE,
        "source_path_feedback_manifest": domain_gap_root / SOURCE_PATH_FEEDBACK_MANIFEST_FILE,
        "source_path_feedback_summary": domain_gap_root / SOURCE_PATH_FEEDBACK_SUMMARY_FILE,
        "source_slices": domain_gap_root / SOURCE_SLICES_FILE,
    }


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "replay_path_feedback_manifest": output_root / REPLAY_PATH_FEEDBACK_MANIFEST_FILE,
        "replay_path_feedback_summary": output_root / REPLAY_PATH_FEEDBACK_SUMMARY_FILE,
        "replay_path_feedback_report": output_root / REPLAY_PATH_FEEDBACK_REPORT_FILE,
        "scenario_results": output_root / SCENARIO_RESULTS_FILE,
        "source_match": output_root / SOURCE_MATCH_AUDIT_FILE,
        "context": output_root / CONTEXT_AUDIT_FILE,
        "boundary": output_root / BOUNDARY_AUDIT_FILE,
        "kill_switch": output_root / KILL_SWITCH_AUDIT_FILE,
        "rollback": output_root / ROLLBACK_AUDIT_FILE,
        "telemetry": output_root / TELEMETRY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _load_evidence(source_paths: dict[str, Path]) -> dict[str, Any]:
    read_reasons: list[str] = []
    return {
        "real_map_preflight": _read_json(source_paths["real_map_preflight_summary"], read_reasons, "real_map_preflight_summary"),
        "domain_gap": _read_json(source_paths["domain_gap_summary"], read_reasons, "domain_gap_summary"),
        "source_path_feedback_manifest": _read_json(
            source_paths["source_path_feedback_manifest"],
            read_reasons,
            "quasi_real_path_feedback_manifest",
        ),
        "source_path_feedback_summary": _read_json(
            source_paths["source_path_feedback_summary"],
            read_reasons,
            "quasi_real_path_feedback_summary",
        ),
        "source_slices": _read_jsonl(source_paths["source_slices"], read_reasons, "quasi_real_map_slices"),
        "read_reason_codes": unique_sorted(read_reasons),
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


def _read_jsonl(path: Path, reasons: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        reasons.append(f"missing_{label}")
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                reasons.append(f"invalid_{label}")
                return []
            rows.append(row)
    except json.JSONDecodeError:
        reasons.append(f"invalid_{label}")
        return []
    return rows


def _build_replay_manifest(source_manifest: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    if not source_manifest:
        return {}
    replay = dict(source_manifest)
    replay["outputs"] = {
        "summary": str(paths["replay_path_feedback_summary"]),
        "report": str(paths["replay_path_feedback_report"]),
    }
    replay["replay_metadata"] = {
        "schema_version": "global-99-real-map-shadow-replay-path-feedback-manifest-metadata/v1",
        "source_outputs": source_manifest.get("outputs", {}),
        "source_manifest_reused": True,
        "contract_and_sidecar_paths_reused": True,
    }
    return replay


def _run_path_feedback_replay(manifest_path: Path, repo_root: Path) -> dict[str, Any]:
    env = dict(os.environ)
    model_src = str(repo_root / "model-explorer" / "src")
    env["PYTHONPATH"] = model_src if not env.get("PYTHONPATH") else model_src + os.pathsep + env["PYTHONPATH"]
    validate = subprocess.run(
        [sys.executable, "-m", "model_explorer", "path-feedback", "validate", str(manifest_path)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if validate.returncode != 0:
        return {
            "status": "failed",
            "reason_codes": ["path_feedback_replay_validate_failed"],
            "validate_exit_code": validate.returncode,
            "run_exit_code": None,
            "validate_stdout": validate.stdout,
            "validate_stderr": validate.stderr,
            "run_stdout": "",
            "run_stderr": "",
        }
    run = subprocess.run(
        [sys.executable, "-m", "model_explorer", "path-feedback", "run", str(manifest_path)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "status": "passed" if run.returncode == 0 else "failed",
        "reason_codes": [] if run.returncode == 0 else ["path_feedback_replay_run_failed"],
        "validate_exit_code": validate.returncode,
        "run_exit_code": run.returncode,
        "validate_stdout": validate.stdout,
        "validate_stderr": validate.stderr,
        "run_stdout": run.stdout,
        "run_stderr": run.stderr,
    }


def _skip_replay(reason_codes: list[str]) -> dict[str, Any]:
    return {
        "status": "skipped" if reason_codes else "not_started",
        "reason_codes": ["path_feedback_replay_skipped_due_to_precondition_failure"] if reason_codes else [],
        "validate_exit_code": None,
        "run_exit_code": None,
    }


def _lineage_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    reasons = list(evidence["read_reason_codes"])
    preflight = evidence["real_map_preflight"]
    domain = evidence["domain_gap"]
    if config["require_preflight_passed"] and preflight.get("status") != "passed":
        reasons.append("real_map_preflight_not_passed")
    if preflight and preflight.get("next_required_change") != "global_99_real_map_shadow_replay":
        reasons.append("real_map_preflight_next_required_change_invalid")
    if domain and domain.get("status") != "passed":
        reasons.append("domain_gap_not_passed")
    return {
        "schema_version": "global-99-real-map-shadow-replay-lineage-audit/v1",
        "lineage_audit_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
    }


def _context_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    slices = evidence["source_slices"]
    domain = evidence["domain_gap"]
    missing = sum(1 for row in slices if not row.get("context_id"))
    legacy = sum(1 for row in slices if row.get("legacy_identity_fallback_used") is True)
    missing = max(missing, _int_from(domain.get("context_id_missing_count"), 0))
    legacy = max(legacy, _int_from(domain.get("legacy_identity_fallback_count"), 0))
    roi_groups = sorted({str(row.get("roi_name") or row.get("scenario_group") or "unknown") for row in slices})
    reasons: list[str] = []
    if config["require_context_ids"] and missing:
        reasons.append("real_map_context_id_missing")
    if config["require_context_ids"] and legacy:
        reasons.append("real_map_legacy_identity_fallback_used")
    if not slices:
        reasons.append("real_map_slice_records_missing")
    return {
        "schema_version": CONTEXT_AUDIT_SCHEMA_VERSION,
        "context_audit_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "slice_count": len(slices) if slices else _int_from(domain.get("slice_count"), 0),
        "roi_group_count": len(roi_groups) if roi_groups else _int_from(domain.get("roi_group_count"), 0),
        "roi_groups": roi_groups,
        "context_id_missing_count": missing,
        "legacy_identity_fallback_count": legacy,
    }


def _source_contract_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    source = evidence["source_path_feedback_summary"]
    domain = evidence["domain_gap"]
    reasons: list[str] = []
    open_grid = bool(source.get("open_grid_fallback_used", False)) or _int_from(domain.get("fallback_or_open_grid_count"), 0) > 0
    safety = _max_int(source.get("tracking_safety_violation_count"), domain.get("safety_regression_count"))
    contract = _max_int(source.get("contract_violation_count"), domain.get("contract_violation_count"))
    path_cost = _max_int(source.get("path_cost_regression_count"), domain.get("path_cost_regression_count"))
    risk = _max_int(source.get("risk_regression_count"), domain.get("risk_regression_count"))
    source_selection = _max_int(source.get("source_selection_regression_count"), domain.get("source_selection_regression_count"))
    if config["require_no_open_grid_fallback"] and open_grid:
        reasons.append("real_map_open_grid_fallback_detected")
    if safety + path_cost + risk + source_selection > 0:
        reasons.append("real_map_path_feedback_regression")
    if contract > 0:
        reasons.append("real_map_contract_violation")
    return {
        "schema_version": "global-99-real-map-shadow-replay-source-contract-audit/v1",
        "source_contract_audit_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "open_grid_fallback_used": open_grid,
        "fallback_or_open_grid_count": _int_from(domain.get("fallback_or_open_grid_count"), 1 if open_grid else 0),
        "safety_regression_count": safety,
        "contract_violation_count": contract,
        "path_cost_regression_count": path_cost,
        "risk_regression_count": risk,
        "source_selection_regression_count": source_selection,
    }


def _source_match_audit(
    config: dict[str, Any],
    evidence: dict[str, Any],
    replay_summary: dict[str, Any],
    replay_run: dict[str, Any],
) -> dict[str, Any]:
    source_summary = evidence["source_path_feedback_summary"]
    reasons = list(replay_run.get("reason_codes", []))
    if replay_run.get("status") not in {"passed", "skipped"} and not replay_summary:
        reasons.append("real_map_shadow_replay_summary_missing")
    if replay_run.get("status") == "skipped":
        reasons.append("real_map_shadow_replay_skipped")
    source_by_id = {row.get("scenario_id"): row for row in source_summary.get("scenarios", []) if row.get("scenario_id")}
    replay_by_id = {row.get("scenario_id"): row for row in replay_summary.get("scenarios", []) if row.get("scenario_id")}
    missing_source = sorted(set(replay_by_id) - set(source_by_id))
    missing_replay = sorted(set(source_by_id) - set(replay_by_id))
    scenario_results = []
    max_coverage_delta = 0.0
    max_path_delta = 0.0
    selected_cell_mismatch_count = 0
    selection_changed_mismatch_count = 0
    scenario_mismatch_count = 0
    scenario_match_count = 0
    for scenario_id in sorted(set(source_by_id) & set(replay_by_id)):
        source = source_by_id[scenario_id]
        replay = replay_by_id[scenario_id]
        coverage_delta = abs(_float_from(source.get("coverage_rate_delta"), 0.0) - _float_from(replay.get("coverage_rate_delta"), 0.0))
        path_delta = abs(
            _float_from(source.get("selected_path_cost_after_feedback"), 0.0)
            - _float_from(replay.get("selected_path_cost_after_feedback"), 0.0)
        )
        selected_cell_match = source.get("selected_cell_after_path_feedback") == replay.get("selected_cell_after_path_feedback")
        selection_changed_match = source.get("selection_changed_by_path_feedback") == replay.get("selection_changed_by_path_feedback")
        coverage_match = coverage_delta <= float(config["source_match_coverage_tolerance"])
        path_match = path_delta <= float(config["source_match_path_cost_tolerance_m"])
        scenario_match = selected_cell_match and selection_changed_match and coverage_match and path_match
        if not scenario_match:
            scenario_mismatch_count += 1
            reasons.append("real_map_shadow_replay_source_mismatch")
        else:
            scenario_match_count += 1
        if not selected_cell_match:
            selected_cell_mismatch_count += 1
        if not selection_changed_match:
            selection_changed_mismatch_count += 1
        max_coverage_delta = max(max_coverage_delta, coverage_delta)
        max_path_delta = max(max_path_delta, path_delta)
        scenario_results.append(
            {
                "schema_version": "global-99-real-map-shadow-replay-scenario-result/v1",
                "scenario_id": scenario_id,
                "scenario_group": source.get("scenario_group", replay.get("scenario_group", "unknown")),
                "source_selected_cell_after_path_feedback": source.get("selected_cell_after_path_feedback"),
                "replay_selected_cell_after_path_feedback": replay.get("selected_cell_after_path_feedback"),
                "selected_cell_match": selected_cell_match,
                "source_selection_changed_by_path_feedback": source.get("selection_changed_by_path_feedback"),
                "replay_selection_changed_by_path_feedback": replay.get("selection_changed_by_path_feedback"),
                "selection_changed_match": selection_changed_match,
                "source_selected_path_cost_after_feedback": _float_from(source.get("selected_path_cost_after_feedback"), 0.0),
                "replay_selected_path_cost_after_feedback": _float_from(replay.get("selected_path_cost_after_feedback"), 0.0),
                "path_cost_delta_m": path_delta,
                "source_coverage_rate_delta": _float_from(source.get("coverage_rate_delta"), 0.0),
                "replay_coverage_rate_delta": _float_from(replay.get("coverage_rate_delta"), 0.0),
                "coverage_delta": coverage_delta,
                "scenario_match": scenario_match,
            }
        )
    if missing_source:
        reasons.append("real_map_shadow_replay_source_has_extra_scenarios")
    if missing_replay:
        reasons.append("real_map_shadow_replay_missing_replay_scenarios")
    scenario_mismatch_count += len(missing_source) + len(missing_replay)
    return {
        "schema_version": SOURCE_MATCH_AUDIT_SCHEMA_VERSION,
        "source_match_audit_passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "source_path_feedback_scenario_count": int(source_summary.get("scenario_count", len(source_by_id))),
        "replay_path_feedback_scenario_count": int(replay_summary.get("scenario_count", len(replay_by_id))) if replay_summary else 0,
        "scenario_match_count": scenario_match_count,
        "scenario_mismatch_count": scenario_mismatch_count,
        "max_replay_coverage_delta": max_coverage_delta,
        "max_replay_path_cost_delta_m": max_path_delta,
        "selection_changed_mismatch_count": selection_changed_mismatch_count,
        "selected_cell_mismatch_count": selected_cell_mismatch_count,
        "missing_source_scenario_ids": missing_source,
        "missing_replay_scenario_ids": missing_replay,
        "scenario_results": scenario_results,
    }


def _boundary_audit(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    violations = []
    for source_name in ("real_map_preflight", "domain_gap", "source_path_feedback_summary"):
        payload = evidence[source_name]
        for field in FORBIDDEN_BOUNDARY_FIELDS:
            if payload.get(field) is True:
                violations.append({"source": source_name, "field": field, "value": True})
        if float(payload.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
            violations.append({"source": source_name, "field": "canary_traffic_fraction", "value": payload.get("canary_traffic_fraction")})
    if float(config["canary_traffic_fraction"]) != 0.0:
        violations.append({"source": "config", "field": "canary_traffic_fraction", "value": config["canary_traffic_fraction"]})
    if not config["offline_path_feedback_replay"]:
        violations.append({"source": "config", "field": "offline_path_feedback_replay", "value": False})
    if not config["allow_offline_path_planner_route_replay"]:
        violations.append({"source": "config", "field": "allow_offline_path_planner_route_replay", "value": False})
    if not config["shadow_replay_mode"]:
        violations.append({"source": "config", "field": "shadow_replay_mode", "value": False})
    return {
        "schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION,
        "boundary_audit_passed": not violations,
        "reason_codes": [] if not violations else ["global_99_real_map_shadow_replay_boundary_violation"],
        "violations": violations,
        **BOUNDARY_FIELDS,
    }


def _kill_switch_audit(source_match: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    passed = boundary["boundary_audit_passed"] and source_match["source_match_audit_passed"]
    return {
        "schema_version": KILL_SWITCH_AUDIT_SCHEMA_VERSION,
        "kill_switch_audit_passed": passed,
        "reason_codes": [] if passed else ["kill_switch_contract_blocked"],
        "kill_switch_required_before_real_map_release_governance": True,
        "kill_switch_real_control_plane_connected": False,
    }


def _rollback_audit(source_match: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    passed = boundary["boundary_audit_passed"] and source_match["source_match_audit_passed"]
    return {
        "schema_version": ROLLBACK_AUDIT_SCHEMA_VERSION,
        "rollback_audit_passed": passed,
        "reason_codes": [] if passed else ["rollback_contract_blocked"],
        "rollback_required_before_real_map_release_governance": True,
        "rollback_real_control_plane_connected": False,
    }


def _telemetry_audit(
    evidence: dict[str, Any],
    replay_summary: dict[str, Any],
    replay_run: dict[str, Any],
    source_match: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    passed = (
        replay_run.get("status") == "passed"
        and bool(replay_summary)
        and source_match["source_path_feedback_scenario_count"] > 0
        and source_match["replay_path_feedback_scenario_count"] > 0
        and context["slice_count"] > 0
    )
    return {
        "schema_version": TELEMETRY_AUDIT_SCHEMA_VERSION,
        "telemetry_audit_passed": passed,
        "reason_codes": [] if passed else ["telemetry_evidence_incomplete"],
        "tracked_counters": {
            "source_path_feedback_scenario_count": source_match["source_path_feedback_scenario_count"],
            "replay_path_feedback_scenario_count": source_match["replay_path_feedback_scenario_count"],
            "slice_count": context["slice_count"],
            "scenario_mismatch_count": source_match["scenario_mismatch_count"],
        },
    }


def _decision(
    evidence: dict[str, Any],
    lineage: dict[str, Any],
    context: dict[str, Any],
    source_contract: dict[str, Any],
    source_match: dict[str, Any],
    replay_run: dict[str, Any],
    boundary: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    reason_codes = unique_sorted(
        list(lineage["reason_codes"])
        + list(context["reason_codes"])
        + list(source_contract["reason_codes"])
        + list(source_match["reason_codes"])
        + list(boundary["reason_codes"])
        + list(kill_switch["reason_codes"])
        + list(rollback["reason_codes"])
        + list(telemetry["reason_codes"])
    )
    read_reasons = set(evidence["read_reason_codes"])
    preflight = evidence["real_map_preflight"]
    if any("real_map_preflight" in reason for reason in read_reasons):
        next_required_change = FIX_PREFLIGHT_NEXT_REQUIRED_CHANGE
    elif preflight.get("status") != "passed" or preflight.get("next_required_change") != "global_99_real_map_shadow_replay":
        next_required_change = FIX_PREFLIGHT_NEXT_REQUIRED_CHANGE
    elif any(
        reason in read_reasons
        for reason in (
            "missing_quasi_real_path_feedback_manifest",
            "invalid_quasi_real_path_feedback_manifest",
            "missing_quasi_real_path_feedback_summary",
            "invalid_quasi_real_path_feedback_summary",
            "missing_quasi_real_map_slices",
            "invalid_quasi_real_map_slices",
            "missing_domain_gap_summary",
            "invalid_domain_gap_summary",
        )
    ):
        next_required_change = FIX_DOMAIN_GAP_NEXT_REQUIRED_CHANGE
    elif replay_run.get("status") == "failed":
        next_required_change = FIX_REPLAY_NEXT_REQUIRED_CHANGE
    elif "real_map_shadow_replay_source_mismatch" in reason_codes:
        next_required_change = FIX_DETERMINISM_NEXT_REQUIRED_CHANGE
    elif "real_map_context_id_missing" in reason_codes or "real_map_legacy_identity_fallback_used" in reason_codes:
        next_required_change = FIX_CONTEXT_NEXT_REQUIRED_CHANGE
    elif (
        "real_map_open_grid_fallback_detected" in reason_codes
        or "real_map_path_feedback_regression" in reason_codes
        or "real_map_contract_violation" in reason_codes
    ):
        next_required_change = FIX_PATH_FEEDBACK_NEXT_REQUIRED_CHANGE
    elif not boundary["boundary_audit_passed"]:
        next_required_change = BOUNDARY_REJECTION_NEXT_REQUIRED_CHANGE
    elif reason_codes:
        next_required_change = FIX_REPLAY_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    passed = not reason_codes and next_required_change == PASS_NEXT_REQUIRED_CHANGE
    return {
        "status": "passed" if passed else "failed",
        "reason_codes": reason_codes,
        "real_map_shadow_replay_passed": passed,
        "real_map_shadow_replay_verdict": PASS_VERDICT if passed else "resolve_global_99_real_map_shadow_replay_rejections",
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
    context: dict[str, Any],
    source_match: dict[str, Any],
    replay_run: dict[str, Any],
    boundary: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    preflight = evidence["real_map_preflight"]
    domain = evidence["domain_gap"]
    source_contract = _source_contract_audit({"require_no_open_grid_fallback": True}, evidence)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "replay_path_feedback_manifest": str(paths["replay_path_feedback_manifest"]),
        "replay_path_feedback_summary": str(paths["replay_path_feedback_summary"]),
        "scenario_results": str(paths["scenario_results"]),
        "source_match_audit": str(paths["source_match"]),
        "context_audit": str(paths["context"]),
        "boundary_audit": str(paths["boundary"]),
        "kill_switch_audit": str(paths["kill_switch"]),
        "rollback_audit": str(paths["rollback"]),
        "telemetry_audit": str(paths["telemetry"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "source_real_map_preflight_status": preflight.get("status", "missing"),
        "source_real_map_preflight_next_required_change": preflight.get("next_required_change", "missing"),
        "source_domain_gap_status": domain.get("status", "missing"),
        "source_path_feedback_scenario_count": source_match["source_path_feedback_scenario_count"],
        "replay_path_feedback_scenario_count": source_match["replay_path_feedback_scenario_count"],
        "slice_count": context["slice_count"],
        "roi_group_count": context["roi_group_count"],
        "context_id_missing_count": context["context_id_missing_count"],
        "legacy_identity_fallback_count": context["legacy_identity_fallback_count"],
        "source_match_audit_passed": source_match["source_match_audit_passed"],
        "scenario_match_count": source_match["scenario_match_count"],
        "scenario_mismatch_count": source_match["scenario_mismatch_count"],
        "max_replay_coverage_delta": source_match["max_replay_coverage_delta"],
        "max_replay_path_cost_delta_m": source_match["max_replay_path_cost_delta_m"],
        "selection_changed_mismatch_count": source_match["selection_changed_mismatch_count"],
        "selected_cell_mismatch_count": source_match["selected_cell_mismatch_count"],
        "open_grid_fallback_used": source_contract["open_grid_fallback_used"],
        "fallback_or_open_grid_count": source_contract["fallback_or_open_grid_count"],
        "safety_regression_count": source_contract["safety_regression_count"],
        "contract_violation_count": source_contract["contract_violation_count"],
        "path_cost_regression_count": source_contract["path_cost_regression_count"],
        "risk_regression_count": source_contract["risk_regression_count"],
        "source_selection_regression_count": source_contract["source_selection_regression_count"],
        "real_map_shadow_replay_passed": decision["real_map_shadow_replay_passed"],
        "real_map_shadow_replay_verdict": decision["real_map_shadow_replay_verdict"],
        "boundary_audit_passed": boundary["boundary_audit_passed"],
        "kill_switch_audit_passed": kill_switch["kill_switch_audit_passed"],
        "rollback_audit_passed": rollback["rollback_audit_passed"],
        "telemetry_audit_passed": telemetry["telemetry_audit_passed"],
        "next_required_change": decision["next_required_change"],
        "replay_run": replay_run,
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
        "stage": "Global 99 Real Map Shadow Replay v1",
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
    context: dict[str, Any],
    source_contract: dict[str, Any],
    source_match: dict[str, Any],
    replay_run: dict[str, Any],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "lineage_reason_codes": lineage["reason_codes"],
        "context_reason_codes": context["reason_codes"],
        "source_contract_reason_codes": source_contract["reason_codes"],
        "source_match_reason_codes": source_match["reason_codes"],
        "replay_run_reason_codes": replay_run.get("reason_codes", []),
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
            "# Global 99 Real Map Shadow Replay v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- verdict: `{summary['real_map_shadow_replay_verdict']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- source_match_audit_passed: `{summary['source_match_audit_passed']}`",
            f"- scenario_mismatch_count: `{summary['scenario_mismatch_count']}`",
            f"- open_grid_fallback_used: `{summary['open_grid_fallback_used']}`",
            f"- boundary_audit_passed: `{summary['boundary_audit_passed']}`",
            "",
            "This stage reruns existing quasi-real path-feedback evidence in an isolated offline shadow replay output root. "
            "It may use the offline path-planner route backend declared by the manifest, but it does not connect a real executor, "
            "start online canary traffic, publish a checkpoint, replace default policy, run PPO, or claim real-world performance.",
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


def _non_negative_float(value: Any, name: str) -> float:
    result = _float_value(value, name)
    if result < 0.0:
        raise ConfigError(f"{name} must be >= 0")
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


def _bool_value(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value


def _float_from(value: Any, default: float) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
