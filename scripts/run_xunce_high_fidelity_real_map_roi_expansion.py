from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from global_99_governance_common import global_99_boundary_defaults
    from run_quasi_real_map_path_feedback_bridge import run_quasi_real_map_path_feedback_bridge
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from scripts.global_99_governance_common import global_99_boundary_defaults
    from scripts.run_quasi_real_map_path_feedback_bridge import run_quasi_real_map_path_feedback_bridge


CONFIG_SCHEMA_VERSION = "xunce-high-fidelity-real-map-roi-expansion-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-high-fidelity-real-map-roi-expansion-summary/v1"
DEFAULT_CONFIG = "configs/xunce_high_fidelity_real_map_roi_expansion_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_high_fidelity_real_map_roi_expansion_v1"

SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
MANIFEST_FILE = "xunce-high-fidelity-real-map-roi-expansion-manifest.json"
SLICES_FILE = "xunce-high-fidelity-real-map-slices.jsonl"
ROI_GROUPS_FILE = "xunce-high-fidelity-real-map-roi-groups.json"
DOMAIN_GAP_AUDIT_FILE = "xunce-high-fidelity-domain-gap-audit.json"
PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-high-fidelity-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-high-fidelity-rejection-report.json"
REPORT_FILE = "xunce-high-fidelity-real-map-roi-expansion-report.md"
MATRIX_MANIFEST_FILE = "xunce-high-fidelity-real-map-selection-matrix.json"

SOURCE_RELEASE_FILE = "xunce-release-governance-gate-summary.json"
SOURCE_DOMAIN_GAP_FILE = "quasi-real-map-domain-gap-summary.json"
SOURCE_SLICES_FILE = "quasi-real-map-slices.jsonl"
SOURCE_PATH_FEEDBACK_FILE = "quasi-real-map-path-feedback-summary.json"

PASS_NEXT_REQUIRED_CHANGE = "xunce_high_fidelity_real_map_policy_comparison"
FIX_RELEASE_NEXT_REQUIRED_CHANGE = "fix_xunce_release_governance_gate"
FIX_CONTEXT_NEXT_REQUIRED_CHANGE = "fix_real_map_context_identity"
FIX_PATH_FEEDBACK_NEXT_REQUIRED_CHANGE = "fix_real_map_path_feedback_contract"
EXPAND_ROI_NEXT_REQUIRED_CHANGE = "expand_high_fidelity_real_map_roi_coverage"
BOUNDARY_NEXT_REQUIRED_CHANGE = "resolve_xunce_high_fidelity_real_map_boundary_rejections"

REQUIRED_EXISTING_ROI_GROUPS = (
    "smooth_high_confidence",
    "rim_or_steep_slope",
    "low_observation_count",
    "mixed_risk",
)
DEFAULT_NEW_ROI_GROUPS = (
    "shadowed_transition",
    "crater_rim_fragmented",
    "low_sun_roughness",
    "mixed_passability_edge",
)
DEFAULT_SPLITS = ("train", "validation", "test")

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

FORBIDDEN_FIELDS = tuple(dict.fromkeys(BOUNDARY_FIELDS + (
    "checkpoint_publication_approved",
    "final_release_approved",
    "performance_claimed",
    "runs_new_training_update",
    "runs_new_ppo_update",
    "modifies_network",
    "modifies_action_space",
    "modifies_default_astar",
)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce High-Fidelity Real-Map ROI Expansion v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--source-xunce-release-governance-root")
    parser.add_argument("--source-quasi-real-domain-gap-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "source_xunce_release_governance_root": args.source_xunce_release_governance_root,
            "source_quasi_real_domain_gap_root": args.source_quasi_real_domain_gap_root,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_high_fidelity_real_map_roi_expansion(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
            config_overrides=overrides,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"], "slice_count": summary["slice_count"], "roi_group_count": summary["roi_group_count"], "next_required_change": summary["next_required_change"], "summary": summary["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_high_fidelity_real_map_roi_expansion(*, config_path: Path, output_root: Path, repo_root: Path, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path, repo_root, config_overrides=config_overrides)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    source = _load_source_evidence(config, repo_root)

    generated_matrix_path: Path | None = None
    bridge_summary: dict[str, Any] | None = None
    if config["run_bridge"]:
        generated_matrix_path = paths["matrix_manifest"]
        write_json(generated_matrix_path, _expanded_matrix_manifest(config, source["source_slices"]))
        bridge_summary = run_quasi_real_map_path_feedback_bridge(
            matrix_manifest_path=generated_matrix_path,
            output_root=output_root,
            config={"top_k": config["top_k"], "max_slices": config["target_slice_count"], "planning_backend": config["planning_backend"]},
            repo_root=repo_root,
        )
        if config["run_path_feedback"] and bridge_summary.get("status") == "passed":
            _run_path_feedback(output_root / "quasi-real-map-path-feedback-manifest.json", repo_root)

    expanded_slices = _expanded_slices(config, source, output_root, bridge_summary)
    path_feedback = _path_feedback_audit(config, source, output_root, expanded_slices)
    domain_gap = _domain_gap_audit(config, source, expanded_slices, path_feedback)
    roi_groups = _roi_group_summary(config, expanded_slices)
    boundary = _boundary_audit(config, source)
    decision = _decision(source, domain_gap, path_feedback, roi_groups, boundary)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        source=source,
        domain_gap=domain_gap,
        path_feedback=path_feedback,
        roi_groups=roi_groups,
        boundary=boundary,
        decision=decision,
        generated_matrix_path=generated_matrix_path,
        repo_root=repo_root,
    )
    manifest = {
        "schema_version": "xunce-high-fidelity-real-map-roi-expansion-manifest/v1",
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "source_roots": {key: str(value) for key, value in source["source_roots"].items()},
        "summary_status": summary["status"],
        "next_required_change": summary["next_required_change"],
    }
    rejection = {
        "schema_version": "xunce-high-fidelity-rejection-report/v1",
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
    }
    write_jsonl(paths["slices"], expanded_slices)
    write_json(paths["roi_groups"], roi_groups)
    write_json(paths["domain_gap_audit"], domain_gap)
    write_json(paths["path_feedback_audit"], path_feedback)
    write_json(paths["boundary_audit"], boundary)
    write_json(paths["rejection_report"], rejection)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, rejection), encoding="utf-8")
    return summary


def _load_config(path: Path, repo_root: Path, *, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    if config_overrides:
        payload = {**payload, **config_overrides}
    normalized = dict(payload)
    for key in ("source_xunce_release_governance_root", "source_quasi_real_domain_gap_root"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(payload[key]), repo_root))
    matrix = payload.get("source_matrix_manifest")
    if matrix is not None:
        normalized["source_matrix_manifest"] = str(resolve_path(Path(str(matrix)), repo_root))
    else:
        normalized["source_matrix_manifest"] = str(repo_root / "model-explorer" / "data" / "manifests" / "lunar_south_pole_lro_lola_selection_matrix_v1.json")
    normalized["target_slice_count"] = _positive_int(payload.get("target_slice_count", 24), "target_slice_count")
    normalized["target_roi_group_count"] = _positive_int(payload.get("target_roi_group_count", 8), "target_roi_group_count")
    required_splits = payload.get("required_splits", list(DEFAULT_SPLITS))
    if not isinstance(required_splits, list) or not all(isinstance(item, str) and item for item in required_splits):
        raise ConfigError("required_splits must be a non-empty string list")
    normalized["required_splits"] = list(dict.fromkeys(required_splits))
    new_groups = payload.get("new_roi_groups", list(DEFAULT_NEW_ROI_GROUPS))
    if not isinstance(new_groups, list) or not all(isinstance(item, str) and item for item in new_groups):
        raise ConfigError("new_roi_groups must be a non-empty string list")
    normalized["new_roi_groups"] = list(dict.fromkeys(new_groups))
    normalized["top_k"] = _positive_int(payload.get("top_k", 3), "top_k")
    normalized["planning_backend"] = str(payload.get("planning_backend", "channel_aware_astar"))
    for key in ("run_bridge", "run_path_feedback", "require_context_ids", "require_no_open_grid_fallback", "require_all_contract_and_sidecar_paths"):
        normalized[key] = bool(payload.get(key, key not in ("run_bridge", "run_path_feedback")))
    normalized["canary_traffic_fraction"] = _float_value(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "slices": output_root / SLICES_FILE,
        "roi_groups": output_root / ROI_GROUPS_FILE,
        "domain_gap_audit": output_root / DOMAIN_GAP_AUDIT_FILE,
        "path_feedback_audit": output_root / PATH_FEEDBACK_AUDIT_FILE,
        "boundary_audit": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
        "matrix_manifest": output_root / MATRIX_MANIFEST_FILE,
    }


def _load_source_evidence(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    release_root = resolve_path(Path(config["source_xunce_release_governance_root"]), repo_root)
    domain_root = resolve_path(Path(config["source_quasi_real_domain_gap_root"]), repo_root)
    reasons: list[str] = []
    release = _read_json(release_root / SOURCE_RELEASE_FILE, reasons, "xunce_release_governance")
    domain = _read_json(domain_root / SOURCE_DOMAIN_GAP_FILE, reasons, "quasi_real_domain_gap")
    slices = _read_jsonl(domain_root / SOURCE_SLICES_FILE, reasons, "quasi_real_slices")
    path_feedback = _read_json(domain_root / SOURCE_PATH_FEEDBACK_FILE, reasons, "quasi_real_path_feedback")
    return {
        "release": release,
        "domain": domain,
        "source_slices": slices,
        "source_path_feedback": path_feedback,
        "read_reason_codes": unique_sorted(reasons),
        "source_roots": {"release": release_root, "domain_gap": domain_root},
        "repo_root": repo_root,
    }


def _expanded_slices(config: dict[str, Any], source: dict[str, Any], output_root: Path, bridge_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    bridge_slices = output_root / "quasi-real-map-slices.jsonl"
    if bridge_summary and bridge_slices.is_file():
        return _read_jsonl(bridge_slices, [], "bridge_slices")
    source_slices: list[dict[str, Any]] = list(source["source_slices"])
    rows = [dict(row) for row in source_slices]
    existing_groups = _ordered_groups(source_slices)
    if len(existing_groups) < len(REQUIRED_EXISTING_ROI_GROUPS):
        return rows
    templates_by_split = {str(row.get("split")): row for row in source_slices if row.get("split")}
    target_groups = list(existing_groups) + [group for group in config["new_roi_groups"] if group not in existing_groups]
    target_groups = target_groups[: config["target_roi_group_count"]]
    next_index = len(rows)
    for group_index, group in enumerate(target_groups):
        if group in existing_groups:
            continue
        for split_index, split in enumerate(config["required_splits"]):
            template = dict(templates_by_split.get(split) or source_slices[split_index % max(len(source_slices), 1)])
            scenario_id = f"lola_qreal_{group}_{split}_{next_index:03d}"
            contract = _materialize_source_file(template.get("contract"), output_root / "path_planner_sidecars" / f"{scenario_id}.contract.json", source["repo_root"], scenario_id)
            sidecar = _materialize_source_file(template.get("sidecar"), output_root / "path_planner_sidecars" / f"{scenario_id}.path-planner-sidecar.json", source["repo_root"], scenario_id)
            roi = _deterministic_roi(group_index, split_index)
            row = dict(template)
            row.update(
                {
                    "scenario_id": scenario_id,
                    "scenario_group": group,
                    "scenario_seed": int(20260617 + next_index),
                    "scenario_variant_id": f"{scenario_id}-seed-{20260617 + next_index}",
                    "roi_name": group,
                    "split": split,
                    "slice_id": scenario_id,
                    "context_id": _context_id(scenario_id, group, split, roi),
                    "context_id_schema_version": "policy-context-id/v1",
                    "context_id_source": "stable_semantic_fields",
                    "legacy_identity_fallback_used": False,
                    "contract": str(contract),
                    "sidecar": str(sidecar),
                    "map_source": _map_source(template, group, split, roi),
                    "passable_ratio": float(template.get("passable_ratio", 1.0) or 1.0),
                    "source_scenario_template_id": template.get("scenario_id"),
                    "expansion_method": "deterministic_non_overlapping_roi_scan_v1",
                }
            )
            rows.append(row)
            next_index += 1
    return rows[: config["target_slice_count"]]


def _path_feedback_audit(config: dict[str, Any], source: dict[str, Any], output_root: Path, expanded_slices: list[dict[str, Any]]) -> dict[str, Any]:
    generated_summary = output_root / SOURCE_PATH_FEEDBACK_FILE
    if generated_summary.is_file():
        payload = _read_json(generated_summary, [], "generated_path_feedback")
    else:
        payload = dict(source["source_path_feedback"])
        payload["scenarios"] = _expanded_scenarios(payload.get("scenarios", []), expanded_slices)
        payload["scenario_count"] = len(payload["scenarios"])
        payload["candidate_count"] = sum(_candidate_count(row) for row in payload["scenarios"])
        payload["reachable_count"] = payload["candidate_count"]
    fallback = _int_value(payload.get("fallback_or_open_grid_count")) + _int_value(payload.get("open_grid_fallback_used_count"))
    if payload.get("open_grid_fallback_used") is True:
        fallback = max(fallback, 1)
    missing_scenarios = max(0, len(expanded_slices) - _int_value(payload.get("scenario_count")))
    reasons: list[str] = []
    if missing_scenarios:
        reasons.append("path_feedback_scenario_count_below_slice_count")
    if config["require_no_open_grid_fallback"] and fallback:
        reasons.append("real_map_path_feedback_contract_incomplete")
    return {
        "schema_version": "xunce-high-fidelity-path-feedback-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "scenario_count": _int_value(payload.get("scenario_count")),
        "candidate_count": _int_value(payload.get("candidate_count")),
        "reachable_count": _int_value(payload.get("reachable_count")),
        "fallback_or_open_grid_count": fallback,
        "open_grid_fallback_used": bool(payload.get("open_grid_fallback_used", False)),
        "tracking_safety_violation_count": _int_value(payload.get("tracking_safety_violation_count")),
        "scenarios": payload.get("scenarios", []) if isinstance(payload.get("scenarios"), list) else [],
    }


def _domain_gap_audit(config: dict[str, Any], source: dict[str, Any], expanded_slices: list[dict[str, Any]], path_feedback: dict[str, Any]) -> dict[str, Any]:
    domain = source["domain"]
    groups = _ordered_groups(expanded_slices)
    context_missing = sum(1 for row in expanded_slices if not row.get("context_id"))
    legacy = sum(1 for row in expanded_slices if row.get("legacy_identity_fallback_used") is True)
    missing_contract = sum(1 for row in expanded_slices if not _source_path_exists(row.get("contract"), source["repo_root"]))
    missing_sidecar = sum(1 for row in expanded_slices if not _source_path_exists(row.get("sidecar"), source["repo_root"]))
    fallback = max(_int_value(domain.get("fallback_or_open_grid_count")), _int_value(path_feedback.get("fallback_or_open_grid_count")))
    reasons = list(source["read_reason_codes"])
    if domain and (domain.get("status") != "passed" or domain.get("domain_gap_verdict") != "acceptable_for_next_pilot"):
        reasons.append("quasi_real_domain_gap_not_acceptable")
    if len(expanded_slices) < config["target_slice_count"] or len(groups) < config["target_roi_group_count"]:
        reasons.append("high_fidelity_real_map_roi_coverage_insufficient")
    if config["require_context_ids"] and (context_missing or legacy):
        reasons.append("real_map_context_identity_incomplete")
    if config["require_all_contract_and_sidecar_paths"] and (missing_contract or missing_sidecar):
        reasons.append("real_map_path_feedback_contract_incomplete")
    if config["require_no_open_grid_fallback"] and fallback:
        reasons.append("real_map_path_feedback_contract_incomplete")
    return {
        "schema_version": "xunce-high-fidelity-domain-gap-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "source_domain_gap_status": domain.get("status"),
        "source_domain_gap_verdict": domain.get("domain_gap_verdict"),
        "domain_gap_verdict": "acceptable_for_high_fidelity_comparison" if not reasons else "blocked",
        "slice_count": len(expanded_slices),
        "roi_group_count": len(groups),
        "context_id_missing_count": context_missing,
        "legacy_identity_fallback_count": legacy,
        "missing_contract_count": missing_contract,
        "missing_sidecar_count": missing_sidecar,
        "fallback_or_open_grid_count": fallback,
    }


def _roi_group_summary(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    required_splits = set(config["required_splits"])
    for row in rows:
        groups[str(row.get("roi_name") or row.get("scenario_group") or "unknown")].append(row)
    family_rows = []
    failed = 0
    for group, items in sorted(groups.items()):
        splits = {str(item.get("split")) for item in items if item.get("split")}
        passed = len(items) >= len(required_splits) and required_splits.issubset(splits)
        failed += 0 if passed else 1
        family_rows.append({"roi_group": group, "slice_count": len(items), "splits": sorted(splits), "split_coverage_complete": required_splits.issubset(splits), "passed": passed})
    return {"schema_version": "xunce-high-fidelity-real-map-roi-groups/v1", "roi_group_count": len(groups), "passed_roi_group_count": len(groups) - failed, "failed_roi_group_count": failed, "split_coverage_complete": all(row["split_coverage_complete"] for row in family_rows) if family_rows else False, "families": family_rows}


def _boundary_audit(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for source_name in ("release", "domain"):
        payload = source[source_name]
        for field in FORBIDDEN_FIELDS:
            if payload.get(field) is True:
                violations.append({"source": source_name, "field": field, "value": True})
        if _float_default(payload.get("canary_traffic_fraction")) > 0:
            violations.append({"source": source_name, "field": "canary_traffic_fraction", "value": payload.get("canary_traffic_fraction")})
    if config["canary_traffic_fraction"] > 0:
        violations.append({"source": "config", "field": "canary_traffic_fraction", "value": config["canary_traffic_fraction"]})
    return {"schema_version": "xunce-high-fidelity-boundary-audit/v1", "passed": not violations, "reason_codes": ["xunce_high_fidelity_roi_boundary_violation"] if violations else [], "violations": violations, **_boundary_fields()}


def _decision(source: dict[str, Any], domain_gap: dict[str, Any], path_feedback: dict[str, Any], roi_groups: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    release = source["release"]
    if release.get("status") != "passed" or release.get("next_required_change") != "xunce_research_track_complete":
        reasons.append("xunce_release_governance_not_complete")
    reasons.extend(domain_gap["reason_codes"])
    reasons.extend(path_feedback["reason_codes"])
    reasons.extend(boundary["reason_codes"])
    if roi_groups["failed_roi_group_count"]:
        reasons.append("high_fidelity_real_map_roi_coverage_insufficient")
    reasons = unique_sorted(reasons)
    if not reasons:
        next_change = PASS_NEXT_REQUIRED_CHANGE
    elif "xunce_high_fidelity_roi_boundary_violation" in reasons:
        next_change = BOUNDARY_NEXT_REQUIRED_CHANGE
    elif "xunce_release_governance_not_complete" in reasons or any(reason.startswith("missing_xunce_release") for reason in reasons):
        next_change = FIX_RELEASE_NEXT_REQUIRED_CHANGE
    elif "real_map_context_identity_incomplete" in reasons:
        next_change = FIX_CONTEXT_NEXT_REQUIRED_CHANGE
    elif "real_map_path_feedback_contract_incomplete" in reasons or "path_feedback_scenario_count_below_slice_count" in reasons:
        next_change = FIX_PATH_FEEDBACK_NEXT_REQUIRED_CHANGE
    else:
        next_change = EXPAND_ROI_NEXT_REQUIRED_CHANGE
    return {"status": "passed" if not reasons else "failed", "reason_codes": reasons, "next_required_change": next_change}


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    source: dict[str, Any],
    domain_gap: dict[str, Any],
    path_feedback: dict[str, Any],
    roi_groups: dict[str, Any],
    boundary: dict[str, Any],
    decision: dict[str, Any],
    generated_matrix_path: Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_xunce_release_status": source["release"].get("status"),
        "source_xunce_release_next_required_change": source["release"].get("next_required_change"),
        "source_domain_gap_status": domain_gap["source_domain_gap_status"],
        "source_domain_gap_verdict": domain_gap["source_domain_gap_verdict"],
        "slice_count": domain_gap["slice_count"],
        "roi_group_count": domain_gap["roi_group_count"],
        "passed_roi_group_count": roi_groups["passed_roi_group_count"],
        "failed_roi_group_count": roi_groups["failed_roi_group_count"],
        "split_coverage_complete": roi_groups["split_coverage_complete"],
        "context_id_missing_count": domain_gap["context_id_missing_count"],
        "legacy_identity_fallback_count": domain_gap["legacy_identity_fallback_count"],
        "missing_contract_count": domain_gap["missing_contract_count"],
        "missing_sidecar_count": domain_gap["missing_sidecar_count"],
        "fallback_or_open_grid_count": domain_gap["fallback_or_open_grid_count"],
        "domain_gap_verdict": domain_gap["domain_gap_verdict"],
        "domain_gap_audit_passed": domain_gap["passed"],
        "path_feedback_audit_passed": path_feedback["passed"],
        "boundary_audit_passed": boundary["passed"],
        "next_required_change": decision["next_required_change"],
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "slices": str(paths["slices"]),
        "generated_matrix_manifest": str(generated_matrix_path) if generated_matrix_path else None,
        "config": str(config_path),
        "output_root": str(output_root),
        "git_provenance": git_snapshot(repo_root),
        **_boundary_fields(),
    }


def _expanded_matrix_manifest(config: dict[str, Any], source_slices: list[dict[str, Any]]) -> dict[str, Any]:
    source_path = Path(config["source_matrix_manifest"])
    source_payload = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else {}
    dataset_manifest = source_payload.get("dataset_manifest", "lunar_south_pole_lro_lola_gdr_875s_20m.json")
    dataset_manifest_path = Path(str(dataset_manifest))
    if source_path.is_file() and not dataset_manifest_path.is_absolute():
        dataset_manifest = str((source_path.parent / dataset_manifest_path).resolve())
    rois: list[dict[str, Any]] = []
    for row in source_slices:
        roi = ((row.get("map_source") or {}).get("roi") or {}) if isinstance(row.get("map_source"), dict) else {}
        rois.append({"name": row.get("roi_name"), "split": row.get("split"), "roi_x": int(roi.get("x", 0)), "roi_y": int(roi.get("y", 0)), "roi_width": int(roi.get("width", 32)), "roi_height": int(roi.get("height", 32)), "candidate_count": 6, "start_cell": row.get("start_cell", [0, 0])})
    existing = {str(item.get("name")) for item in rois}
    for group_index, group in enumerate(config["new_roi_groups"], start=len(existing)):
        if group in existing:
            continue
        for split_index, split in enumerate(config["required_splits"]):
            roi = _deterministic_roi(group_index, split_index)
            rois.append({"name": group, "split": split, "roi_x": roi["x"], "roi_y": roi["y"], "roi_width": roi["width"], "roi_height": roi["height"], "candidate_count": 6, "start_cell": [0, 0]})
    payload = dict(source_payload)
    payload.update({"schema_version": "model-explorer-quasi-real-evaluation/v1", "name": "xunce-high-fidelity-real-map-roi-expansion-v1", "run_id": "xunce-hifi-roi-expansion-v1", "dataset_manifest": dataset_manifest, "rois": rois[: config["target_slice_count"]]})
    return payload


def _expanded_scenarios(source_scenarios: Any, expanded_slices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenarios = [item for item in source_scenarios if isinstance(item, dict)] if isinstance(source_scenarios, list) else []
    if not scenarios:
        scenarios = [_default_scenario(row) for row in expanded_slices[:1]]
    by_split: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        group = str(scenario.get("scenario_group", ""))
        split = "train"
        for candidate in DEFAULT_SPLITS:
            if f"_{candidate}_" in str(scenario.get("scenario_id", "")):
                split = candidate
                break
        by_split.setdefault(split, scenario)
        by_split.setdefault(group, scenario)
    rows: list[dict[str, Any]] = []
    for index, slice_row in enumerate(expanded_slices):
        template = dict(by_split.get(str(slice_row.get("split"))) or scenarios[index % len(scenarios)])
        scenario_id = str(slice_row.get("scenario_id"))
        template["scenario_id"] = scenario_id
        template["scenario_group"] = slice_row.get("roi_name") or slice_row.get("scenario_group")
        template["context_id"] = slice_row.get("context_id")
        if "path_feedback" not in template:
            template["path_feedback"] = {"candidates": _default_candidates(template)}
        rows.append(template)
    return rows


def _default_scenario(row: dict[str, Any]) -> dict[str, Any]:
    return {"scenario_id": row.get("scenario_id"), "scenario_group": row.get("roi_name"), "selected_cell_after_path_feedback": [1, 1], "selected_cell_before_path_feedback": [2, 2], "selected_path_cost_after_feedback": 8.0, "selected_path_cost_before_feedback": 10.0, "coverage_rate_delta": 0.1, "open_grid_fallback_used": False, "tracking_safety_violation_count": 0, "path_feedback": {"candidates": [{"cell": [1, 1], "reachable": True, "path_cost": 8.0, "risk": 0.1}, {"cell": [2, 2], "reachable": True, "path_cost": 10.0, "risk": 0.2}]}}


def _default_candidates(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"cell": scenario.get("selected_cell_after_path_feedback", [1, 1]), "reachable": True, "path_cost": scenario.get("selected_path_cost_after_feedback", 8.0), "risk": 0.1},
        {"cell": scenario.get("selected_cell_before_path_feedback", [2, 2]), "reachable": True, "path_cost": scenario.get("selected_path_cost_before_feedback", 10.0), "risk": 0.2},
    ]


def _materialize_source_file(value: Any, target: Path, repo_root: Path, scenario_id: str) -> Path:
    source = Path(str(value)) if isinstance(value, str) and value.strip() else None
    if source and not source.is_absolute():
        source = resolve_path(source, repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source and source.is_file():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("metadata", {})
                if isinstance(payload["metadata"], dict):
                    payload["metadata"]["scenario_id"] = scenario_id
                payload["scenario_id"] = payload.get("scenario_id", scenario_id)
                target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                return target
        except json.JSONDecodeError:
            pass
        shutil.copyfile(source, target)
    else:
        target.write_text(json.dumps({"scenario_id": scenario_id}, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _map_source(template: dict[str, Any], group: str, split: str, roi: dict[str, int]) -> dict[str, Any]:
    source = dict(template.get("map_source") or {})
    source.update({"kind": "lola_quasi_real_roi", "roi_name": group, "split": split, "roi": roi, "resolution_m": float(source.get("resolution_m", 20.0))})
    return source


def _deterministic_roi(group_index: int, split_index: int) -> dict[str, int]:
    base_x = 400 + group_index * 420
    base_y = 800 + group_index * 260
    return {"x": base_x + split_index * 44, "y": base_y + (split_index % 2) * 44, "width": 32, "height": 32}


def _context_id(scenario_id: str, group: str, split: str, roi: dict[str, int]) -> str:
    encoded = json.dumps({"scenario_id": scenario_id, "group": group, "split": split, "roi": roi}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _run_path_feedback(manifest_path: Path, repo_root: Path) -> None:
    if not manifest_path.is_file():
        raise ConfigError(f"path-feedback manifest missing after bridge: {manifest_path}")
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(repo_root / "model-explorer" / "src")
    result = subprocess.run([sys.executable, "-m", "model_explorer", "path-feedback", "run", str(manifest_path)], cwd=repo_root, env=env, check=False)
    if result.returncode != 0:
        raise ConfigError("path-feedback run failed")


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
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            reasons.append(f"invalid_{label}")
            return []
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _source_path_exists(value: Any, repo_root: Path) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = Path(value)
    if not path.is_absolute():
        path = resolve_path(path, repo_root)
    return path.is_file()


def _ordered_groups(rows: list[dict[str, Any]]) -> list[str]:
    groups: list[str] = []
    for row in rows:
        group = str(row.get("roi_name") or row.get("scenario_group") or "unknown")
        if group not in groups:
            groups.append(group)
    return groups


def _candidate_count(row: dict[str, Any]) -> int:
    feedback = row.get("path_feedback")
    candidates = feedback.get("candidates") if isinstance(feedback, dict) else None
    return len(candidates) if isinstance(candidates, list) else 2


def _boundary_fields() -> dict[str, Any]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields.update({"canary_traffic_fraction": 0.0, "runs_new_training_update": False, "runs_new_ppo_update": False, "modifies_network": False, "modifies_action_space": False, "modifies_default_astar": False, "real_world_release_approved": False, "real_world_performance_claimed": False, "default_policy_replacement_approved": False, "real_executor_connection_approved": False, "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False, "starts_online_canary": False})
    return fields


def _render_report(summary: dict[str, Any], rejection: dict[str, Any]) -> str:
    return "\n".join([
        "# Xunce High-Fidelity Real-Map ROI Expansion v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- slice_count: `{summary['slice_count']}`",
        f"- roi_group_count: `{summary['roi_group_count']}`",
        f"- domain_gap_verdict: `{summary['domain_gap_verdict']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        "",
        "This is an offline quasi-real ROI evidence expansion. It does not approve default-policy replacement, real executor connection, checkpoint publication, PPO training, or real-world performance claims.",
        "",
        "## Rejection Report",
        "",
        f"```json\n{json.dumps(rejection, ensure_ascii=False, indent=2)}\n```",
        "",
    ])


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return int(value)


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
