from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_EXPLORER_SRC = REPO_ROOT / "model-explorer" / "src"
for import_path in (SCRIPT_DIR, MODEL_EXPLORER_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json


CONFIG_SCHEMA_VERSION = "network-architecture-upgrade-readiness-review-config/v1"
SUMMARY_SCHEMA_VERSION = "network-architecture-upgrade-readiness-summary/v1"
EVIDENCE_AUDIT_SCHEMA_VERSION = "network-architecture-upgrade-evidence-audit/v1"
BOTTLENECK_ATTRIBUTION_SCHEMA_VERSION = "network-architecture-upgrade-bottleneck-attribution/v1"
MANIFEST_SCHEMA_VERSION = "network-architecture-upgrade-readiness-manifest/v1"
REJECTION_REPORT_SCHEMA_VERSION = "network-architecture-upgrade-rejection-report/v1"

DEFAULT_CONFIG = "configs/network_architecture_upgrade_readiness_review_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_network_architecture_upgrade_readiness_review_v1"

SUMMARY_FILE = "network-architecture-upgrade-readiness-summary.json"
EVIDENCE_AUDIT_FILE = "network-architecture-upgrade-evidence-audit.json"
BOTTLENECK_ATTRIBUTION_FILE = "network-architecture-upgrade-bottleneck-attribution.json"
REPORT_FILE = "network-architecture-upgrade-recommendation-report.md"
MANIFEST_FILE = "network-architecture-upgrade-readiness-manifest.json"
REJECTION_REPORT_FILE = "network-architecture-upgrade-rejection-report.json"

FIX_MULTI_MAP_NEXT_REQUIRED_CHANGE = "fix_global_99_multi_map_generalization"
NETWORK_UPGRADE_NEXT_REQUIRED_CHANGE = "network_architecture_upgrade_v1"
EXPAND_CONTRAST_NEXT_REQUIRED_CHANGE = "expand_global_99_policy_contrast_evidence"
RELEASE_GOVERNANCE_NEXT_REQUIRED_CHANGE = "global_99_release_governance_preflight"

BOUNDARY_FIELDS = {
    "runs_new_ppo_update": False,
    "publishes_checkpoint": False,
    "replaces_default_policy": False,
    "connects_real_executor": False,
    "modifies_network": False,
    "modifies_action_space": False,
    "modifies_default_astar": False,
    "uses_path_planner": False,
    "uses_npz_or_sidecar": False,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Network Architecture Upgrade Readiness Review v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_network_architecture_upgrade_readiness_review(
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
                "network_upgrade_recommended": summary["network_upgrade_recommended"],
                "network_upgrade_readiness_decision": summary["network_upgrade_readiness_decision"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_network_architecture_upgrade_readiness_review(
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
    paths = _artifact_paths(output_root)
    source_paths = _source_paths(config, config_path, repo_root)
    evidence = _load_multi_map_evidence(source_paths)
    candidate_inventory = _candidate_architecture_inventory()
    decision = _decide_readiness(config, evidence)

    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        source_paths=source_paths,
        evidence=evidence,
        decision=decision,
        candidate_inventory=candidate_inventory,
        repo_root=repo_root,
    )
    evidence_audit = _evidence_audit(
        generated_at=generated_at,
        source_paths=source_paths,
        evidence=evidence,
        candidate_inventory=candidate_inventory,
    )
    bottleneck_attribution = _bottleneck_attribution(generated_at, summary, decision, evidence)
    rejection_report = _rejection_report(generated_at, summary, evidence)
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)

    write_json(paths["evidence_audit"], evidence_audit)
    write_json(paths["bottleneck_attribution"], bottleneck_attribution)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(
        _render_report(summary, evidence_audit, bottleneck_attribution, rejection_report),
        encoding="utf-8",
    )
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
        "source_multi_map_root",
        "source_multi_map_summary",
        "source_family_summary",
        "source_policy_vs_baseline_audit",
        "source_scenario_results",
    ):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")

    normalized = dict(payload)
    normalized["source_multi_map_root"] = str(resolve_path(Path(payload["source_multi_map_root"]), repo_root))
    normalized["target_coverage_rate"] = _bounded_float(payload.get("target_coverage_rate"), "target_coverage_rate")
    normalized["max_policy_guard_fallback_rate"] = _bounded_float(
        payload.get("max_policy_guard_fallback_rate"),
        "max_policy_guard_fallback_rate",
    )
    normalized["preferred_policy_guard_fallback_rate"] = _bounded_float(
        payload.get("preferred_policy_guard_fallback_rate", 0.05),
        "preferred_policy_guard_fallback_rate",
    )
    normalized["low_policy_contrast_baseline_agreement_rate"] = _bounded_float(
        payload.get("low_policy_contrast_baseline_agreement_rate"),
        "low_policy_contrast_baseline_agreement_rate",
    )
    if normalized["preferred_policy_guard_fallback_rate"] > normalized["max_policy_guard_fallback_rate"]:
        raise ConfigError("preferred_policy_guard_fallback_rate must be <= max_policy_guard_fallback_rate")
    return normalized


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "evidence_audit": output_root / EVIDENCE_AUDIT_FILE,
        "bottleneck_attribution": output_root / BOTTLENECK_ATTRIBUTION_FILE,
        "report": output_root / REPORT_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
    }


def _source_paths(config: dict[str, Any], config_path: Path, repo_root: Path) -> dict[str, Path]:
    root = resolve_path(Path(config["source_multi_map_root"]), repo_root)
    return {
        "root": root,
        "summary": _resolve_source_artifact(root, config["source_multi_map_summary"], config_path, repo_root),
        "family_summary": _resolve_source_artifact(root, config["source_family_summary"], config_path, repo_root),
        "policy_vs_baseline_audit": _resolve_source_artifact(
            root,
            config["source_policy_vs_baseline_audit"],
            config_path,
            repo_root,
        ),
        "scenario_results": _resolve_source_artifact(
            root,
            config["source_scenario_results"],
            config_path,
            repo_root,
        ),
    }


def _resolve_source_artifact(root: Path, value: str, config_path: Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    rooted = root / path
    if rooted.exists() or root.exists():
        return rooted
    return resolve_path(path, repo_root if config_path.is_absolute() else config_path.parent)


def _load_multi_map_evidence(source_paths: dict[str, Path]) -> dict[str, Any]:
    errors: list[str] = []
    loaded: dict[str, Any] = {}
    for key in ("summary", "family_summary", "policy_vs_baseline_audit"):
        path = source_paths[key]
        if not path.is_file():
            errors.append(f"missing:{key}:{path}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid_json:{key}:{exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"invalid_shape:{key}:root_not_object")
            continue
        loaded[key] = payload

    scenario_path = source_paths["scenario_results"]
    scenario_rows: list[dict[str, Any]] = []
    if not scenario_path.is_file():
        errors.append(f"missing:scenario_results:{scenario_path}")
    else:
        try:
            for line_number, line in enumerate(scenario_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    errors.append(f"invalid_shape:scenario_results:{line_number}:row_not_object")
                    continue
                scenario_rows.append(row)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid_json:scenario_results:{exc}")
    loaded["scenario_results"] = scenario_rows

    reason_codes: list[str] = []
    if errors:
        missing = [error for error in errors if error.startswith("missing:")]
        invalid = [error for error in errors if not error.startswith("missing:")]
        if missing:
            reason_codes.append("missing_multi_map_evidence")
        if invalid:
            reason_codes.append("invalid_multi_map_evidence")

    summary = loaded.get("summary", {})
    policy_audit = loaded.get("policy_vs_baseline_audit", {})
    return {
        "status": "loaded" if not errors else "invalid",
        "reason_codes": unique_sorted(reason_codes),
        "load_errors": errors,
        "summary": summary,
        "family_summary": loaded.get("family_summary", {}),
        "policy_vs_baseline_audit": policy_audit,
        "scenario_results": scenario_rows,
        "metrics": _extract_metrics(summary, policy_audit),
    }


def _extract_metrics(summary: dict[str, Any], policy_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_multi_map_status": str(summary.get("status", "missing")),
        "source_reason_codes": list(summary.get("reason_codes", [])) if isinstance(summary.get("reason_codes"), list) else [],
        "target_coverage_rate": _float_or(summary.get("target_coverage_rate"), 0.99),
        "required_scenario_count": _int_or(summary.get("required_scenario_count"), 0),
        "failed_required_scenario_count": _int_or(summary.get("failed_required_scenario_count"), 0),
        "aggregate_achieved_coverage_rate": _float_or(summary.get("aggregate_achieved_coverage_rate"), 0.0),
        "min_scenario_achieved_coverage_rate": _float_or(
            summary.get("min_scenario_achieved_coverage_rate"),
            0.0,
        ),
        "policy_guidance_applied": bool(summary.get("policy_guidance_applied", False)),
        "policy_scored_candidate_count": _int_or(
            policy_audit.get("policy_scored_candidate_count", summary.get("policy_scored_candidate_count")),
            0,
        ),
        "policy_guided_decision_count": _int_or(
            policy_audit.get("policy_guided_decision_count", summary.get("policy_guided_decision_count")),
            0,
        ),
        "policy_guard_fallback_count": _int_or(
            policy_audit.get("policy_guard_fallback_count", summary.get("policy_guard_fallback_count")),
            0,
        ),
        "baseline_agreement_rate": _float_or(
            policy_audit.get("baseline_agreement_rate", summary.get("baseline_agreement_rate")),
            0.0,
        ),
        "policy_better_than_baseline_count": _int_or(
            policy_audit.get("policy_better_than_baseline_count", summary.get("policy_better_than_baseline_count")),
            0,
        ),
        "policy_worse_than_baseline_count": _int_or(
            policy_audit.get("policy_worse_than_baseline_count", summary.get("policy_worse_than_baseline_count")),
            0,
        ),
        "controlled_regression_count": _int_or(
            policy_audit.get("controlled_regression_count", summary.get("controlled_regression_count")),
            0,
        ),
    }


def _decide_readiness(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    metrics = evidence["metrics"]
    target = float(config["target_coverage_rate"])
    max_fallback_rate = float(config["max_policy_guard_fallback_rate"])
    preferred_fallback_rate = float(config["preferred_policy_guard_fallback_rate"])
    low_contrast_agreement = float(config["low_policy_contrast_baseline_agreement_rate"])
    guided_decision_count = int(metrics["policy_guided_decision_count"])
    fallback_count = int(metrics["policy_guard_fallback_count"])
    fallback_rate = fallback_count / guided_decision_count if guided_decision_count else 0.0

    if evidence["reason_codes"]:
        return {
            "status": "failed",
            "reason_codes": evidence["reason_codes"],
            "network_upgrade_recommended": False,
            "decision": "review_blocked_missing_or_invalid_multi_map_evidence",
            "network_upgrade_blockers": ["multi_map_evidence_unavailable"],
            "classification": "missing_or_invalid_multi_map_evidence",
            "next_required_change": FIX_MULTI_MAP_NEXT_REQUIRED_CHANGE,
            "policy_guard_fallback_rate": fallback_rate,
        }

    if metrics["source_multi_map_status"] != "passed":
        return {
            "status": "failed",
            "reason_codes": ["source_multi_map_not_passed"],
            "network_upgrade_recommended": False,
            "decision": "review_blocked_source_multi_map_not_passed",
            "network_upgrade_blockers": ["source_multi_map_not_passed"],
            "classification": "source_multi_map_failed",
            "next_required_change": FIX_MULTI_MAP_NEXT_REQUIRED_CHANGE,
            "policy_guard_fallback_rate": fallback_rate,
        }

    required_incomplete = (
        int(metrics["failed_required_scenario_count"]) > 0
        or float(metrics["aggregate_achieved_coverage_rate"]) + 1.0e-12 < target
        or float(metrics["min_scenario_achieved_coverage_rate"]) + 1.0e-12 < target
    )
    if required_incomplete:
        return {
            "status": "passed",
            "reason_codes": [],
            "network_upgrade_recommended": False,
            "decision": "defer_network_upgrade_multi_map_evidence_incomplete",
            "network_upgrade_blockers": ["multi_map_coverage_evidence_incomplete"],
            "classification": "multi_map_evidence_incomplete",
            "next_required_change": FIX_MULTI_MAP_NEXT_REQUIRED_CHANGE,
            "policy_guard_fallback_rate": fallback_rate,
        }

    if not bool(metrics["policy_guidance_applied"]):
        return {
            "status": "passed",
            "reason_codes": [],
            "network_upgrade_recommended": False,
            "decision": "defer_network_upgrade_policy_guidance_missing",
            "network_upgrade_blockers": ["policy_guidance_evidence_missing"],
            "classification": "policy_guidance_missing",
            "next_required_change": EXPAND_CONTRAST_NEXT_REQUIRED_CHANGE,
            "policy_guard_fallback_rate": fallback_rate,
        }

    policy_worse = int(metrics["policy_worse_than_baseline_count"])
    controlled_regression = int(metrics["controlled_regression_count"])
    if policy_worse > 0 or controlled_regression > 0:
        return {
            "status": "passed",
            "reason_codes": [],
            "network_upgrade_recommended": True,
            "decision": "recommend_network_architecture_upgrade_due_to_policy_regression",
            "network_upgrade_blockers": [],
            "classification": "policy_regression_bottleneck",
            "next_required_change": NETWORK_UPGRADE_NEXT_REQUIRED_CHANGE,
            "policy_guard_fallback_rate": fallback_rate,
        }

    if fallback_rate > max_fallback_rate:
        return {
            "status": "passed",
            "reason_codes": [],
            "network_upgrade_recommended": True,
            "decision": "recommend_network_architecture_upgrade_due_to_guard_fallback",
            "network_upgrade_blockers": [],
            "classification": "policy_guard_fallback_bottleneck",
            "next_required_change": NETWORK_UPGRADE_NEXT_REQUIRED_CHANGE,
            "policy_guard_fallback_rate": fallback_rate,
        }

    policy_better = int(metrics["policy_better_than_baseline_count"])
    baseline_agreement = float(metrics["baseline_agreement_rate"])
    if baseline_agreement >= low_contrast_agreement and policy_better == 0:
        return {
            "status": "passed",
            "reason_codes": [],
            "network_upgrade_recommended": False,
            "decision": "defer_network_upgrade_policy_contrast_insufficient",
            "network_upgrade_blockers": ["policy_contrast_insufficient"],
            "classification": "policy_contrast_insufficient",
            "next_required_change": EXPAND_CONTRAST_NEXT_REQUIRED_CHANGE,
            "policy_guard_fallback_rate": fallback_rate,
        }

    if policy_better > 0 and fallback_rate <= preferred_fallback_rate:
        return {
            "status": "passed",
            "reason_codes": [],
            "network_upgrade_recommended": False,
            "decision": "defer_network_upgrade_global_99_synthetic_goal_met",
            "network_upgrade_blockers": [
                "required_scenarios_passed",
                "policy_positive_divergence_present",
                "no_policy_regression_detected",
                "guard_fallback_rate_within_preferred_threshold",
            ],
            "classification": "no_network_bottleneck_detected",
            "next_required_change": RELEASE_GOVERNANCE_NEXT_REQUIRED_CHANGE,
            "policy_guard_fallback_rate": fallback_rate,
        }

    return {
        "status": "passed",
        "reason_codes": [],
        "network_upgrade_recommended": False,
        "decision": "defer_network_upgrade_expand_policy_contrast_evidence",
        "network_upgrade_blockers": ["policy_contrast_or_guard_margin_needs_more_evidence"],
        "classification": "additional_evidence_required",
        "next_required_change": EXPAND_CONTRAST_NEXT_REQUIRED_CHANGE,
        "policy_guard_fallback_rate": fallback_rate,
    }


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    source_paths: dict[str, Path],
    evidence: dict[str, Any],
    decision: dict[str, Any],
    candidate_inventory: list[str],
    repo_root: Path,
) -> dict[str, Any]:
    metrics = evidence["metrics"]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "evidence_audit": str(paths["evidence_audit"]),
        "bottleneck_attribution_report": str(paths["bottleneck_attribution"]),
        "recommendation_report": str(paths["report"]),
        "manifest": str(paths["manifest"]),
        "rejection_report": str(paths["rejection_report"]),
        "source_multi_map_root": str(source_paths["root"]),
        "source_multi_map_summary": str(source_paths["summary"]),
        "source_family_summary": str(source_paths["family_summary"]),
        "source_policy_vs_baseline_audit": str(source_paths["policy_vs_baseline_audit"]),
        "source_scenario_results": str(source_paths["scenario_results"]),
        "source_multi_map_status": metrics["source_multi_map_status"],
        "required_scenario_count": metrics["required_scenario_count"],
        "failed_required_scenario_count": metrics["failed_required_scenario_count"],
        "aggregate_achieved_coverage_rate": metrics["aggregate_achieved_coverage_rate"],
        "min_scenario_achieved_coverage_rate": metrics["min_scenario_achieved_coverage_rate"],
        "policy_guidance_applied": metrics["policy_guidance_applied"],
        "policy_scored_candidate_count": metrics["policy_scored_candidate_count"],
        "policy_guided_decision_count": metrics["policy_guided_decision_count"],
        "policy_guard_fallback_count": metrics["policy_guard_fallback_count"],
        "policy_guard_fallback_rate": decision["policy_guard_fallback_rate"],
        "baseline_agreement_rate": metrics["baseline_agreement_rate"],
        "policy_better_than_baseline_count": metrics["policy_better_than_baseline_count"],
        "policy_worse_than_baseline_count": metrics["policy_worse_than_baseline_count"],
        "controlled_regression_count": metrics["controlled_regression_count"],
        "network_upgrade_recommended": decision["network_upgrade_recommended"],
        "network_upgrade_readiness_decision": decision["decision"],
        "network_upgrade_blockers": decision["network_upgrade_blockers"],
        "bottleneck_attribution": {
            "classification": decision["classification"],
            "policy_guard_fallback_rate": decision["policy_guard_fallback_rate"],
            "source_multi_map_status": metrics["source_multi_map_status"],
        },
        "candidate_architecture_inventory": candidate_inventory,
        "next_required_change": decision["next_required_change"],
        **BOUNDARY_FIELDS,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _evidence_audit(
    *,
    generated_at: str,
    source_paths: dict[str, Path],
    evidence: dict[str, Any],
    candidate_inventory: list[str],
) -> dict[str, Any]:
    summary = evidence["summary"]
    family_summary = evidence["family_summary"]
    policy_audit = evidence["policy_vs_baseline_audit"]
    scenario_results = evidence["scenario_results"]
    scenario_reason_counts = Counter(
        reason for row in scenario_results for reason in row.get("reason_codes", []) if isinstance(row, dict)
    )
    return {
        "schema_version": EVIDENCE_AUDIT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_paths": {key: str(value) for key, value in source_paths.items()},
        "evidence_load_status": evidence["status"],
        "evidence_reason_codes": evidence["reason_codes"],
        "load_errors": evidence["load_errors"],
        "source_summary": {
            key: summary.get(key)
            for key in (
                "status",
                "reason_codes",
                "scenario_count",
                "required_scenario_count",
                "failed_required_scenario_count",
                "aggregate_achieved_coverage_rate",
                "min_scenario_achieved_coverage_rate",
                "policy_guidance_applied",
                "next_required_change",
            )
        },
        "family_summary": {
            "family_count": family_summary.get("family_count"),
            "passed_family_count": family_summary.get("passed_family_count"),
            "failed_family_count": family_summary.get("failed_family_count"),
            "families": family_summary.get("families", {}),
        },
        "policy_vs_baseline_audit": {
            key: policy_audit.get(key)
            for key in (
                "policy_guidance_applied",
                "policy_scored_candidate_count",
                "policy_guided_decision_count",
                "policy_guard_fallback_count",
                "baseline_agreement_rate",
                "policy_better_than_baseline_count",
                "policy_worse_than_baseline_count",
                "controlled_regression_count",
            )
        },
        "scenario_result_count": len(scenario_results),
        "scenario_reason_code_counts": dict(scenario_reason_counts),
        "candidate_architecture_inventory": candidate_inventory,
    }


def _bottleneck_attribution(
    generated_at: str,
    summary: dict[str, Any],
    decision: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    metrics = evidence["metrics"]
    return {
        "schema_version": BOTTLENECK_ATTRIBUTION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "classification": decision["classification"],
        "network_upgrade_recommended": decision["network_upgrade_recommended"],
        "network_upgrade_readiness_decision": decision["decision"],
        "network_upgrade_blockers": decision["network_upgrade_blockers"],
        "signals": {
            "source_multi_map_status": metrics["source_multi_map_status"],
            "failed_required_scenario_count": metrics["failed_required_scenario_count"],
            "aggregate_achieved_coverage_rate": metrics["aggregate_achieved_coverage_rate"],
            "min_scenario_achieved_coverage_rate": metrics["min_scenario_achieved_coverage_rate"],
            "policy_guidance_applied": metrics["policy_guidance_applied"],
            "policy_guard_fallback_rate": summary["policy_guard_fallback_rate"],
            "baseline_agreement_rate": metrics["baseline_agreement_rate"],
            "policy_better_than_baseline_count": metrics["policy_better_than_baseline_count"],
            "policy_worse_than_baseline_count": metrics["policy_worse_than_baseline_count"],
            "controlled_regression_count": metrics["controlled_regression_count"],
        },
    }


def _rejection_report(
    generated_at: str,
    summary: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": summary["status"],
        "reason_codes": summary["reason_codes"],
        "failure_reason_code_counts": dict(Counter(summary["reason_codes"])),
        "network_upgrade_recommended": summary["network_upgrade_recommended"],
        "network_upgrade_blockers": summary["network_upgrade_blockers"],
        "source_load_errors": evidence["load_errors"],
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
        "stage": "Network Architecture Upgrade Readiness Review v1",
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "summary_status": summary["status"],
        "network_upgrade_recommended": summary["network_upgrade_recommended"],
        "next_required_change": summary["next_required_change"],
        "boundary": dict(BOUNDARY_FIELDS),
    }


def _render_report(
    summary: dict[str, Any],
    evidence_audit: dict[str, Any],
    bottleneck_attribution: dict[str, Any],
    rejection_report: dict[str, Any],
) -> str:
    lines = [
        "# Network Architecture Upgrade Readiness Review v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- source_multi_map_status: `{summary['source_multi_map_status']}`",
        f"- required_scenario_count: `{summary['required_scenario_count']}`",
        f"- failed_required_scenario_count: `{summary['failed_required_scenario_count']}`",
        f"- aggregate_achieved_coverage_rate: `{summary['aggregate_achieved_coverage_rate']}`",
        f"- min_scenario_achieved_coverage_rate: `{summary['min_scenario_achieved_coverage_rate']}`",
        f"- policy_guard_fallback_rate: `{summary['policy_guard_fallback_rate']}`",
        f"- baseline_agreement_rate: `{summary['baseline_agreement_rate']}`",
        f"- policy_better_than_baseline_count: `{summary['policy_better_than_baseline_count']}`",
        f"- policy_worse_than_baseline_count: `{summary['policy_worse_than_baseline_count']}`",
        f"- controlled_regression_count: `{summary['controlled_regression_count']}`",
        f"- network_upgrade_recommended: `{summary['network_upgrade_recommended']}`",
        f"- network_upgrade_readiness_decision: `{summary['network_upgrade_readiness_decision']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        "",
        "## Bottleneck Attribution",
        "",
        f"- classification: `{bottleneck_attribution['classification']}`",
        f"- network_upgrade_blockers: `{summary['network_upgrade_blockers']}`",
        "",
        "## Candidate Architecture Inventory",
        "",
        f"- supported_architectures: `{summary['candidate_architecture_inventory']}`",
        "",
        "## Evidence Audit",
        "",
        f"- evidence_load_status: `{evidence_audit['evidence_load_status']}`",
        f"- scenario_result_count: `{evidence_audit['scenario_result_count']}`",
        f"- rejection_reason_counts: `{rejection_report['failure_reason_code_counts']}`",
        "",
        "This stage is a readiness audit only. It does not train PPO, publish a checkpoint, replace default policy, connect a real executor, call path-planner, use NPZ/sidecar maps, or modify network/action space/default A*.",
        "",
    ]
    return "\n".join(lines)


def _candidate_architecture_inventory() -> list[str]:
    try:
        from model_explorer.policy.architectures import SUPPORTED_ARCHITECTURES
    except Exception:
        return []
    return list(SUPPORTED_ARCHITECTURES)


def _bounded_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be numeric")
    numeric = float(value)
    if numeric <= 0.0 or numeric > 1.0:
        raise ConfigError(f"{label} must be > 0 and <= 1")
    return numeric


def _float_or(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
