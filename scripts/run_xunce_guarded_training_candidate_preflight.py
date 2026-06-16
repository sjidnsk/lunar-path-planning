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
    from global_99_governance_common import global_99_boundary_defaults
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json
    from scripts.global_99_governance_common import global_99_boundary_defaults


CONFIG_SCHEMA_VERSION = "xunce-guarded-training-candidate-preflight-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-guarded-training-candidate-preflight-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-guarded-training-candidate-preflight-manifest/v1"
SOURCE_AUDIT_SCHEMA_VERSION = "xunce-guarded-training-source-evidence-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-guarded-training-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-guarded-training-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_guarded_training_candidate_preflight_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_guarded_training_candidate_preflight_v1"

SUMMARY_FILE = "xunce-guarded-training-candidate-preflight-summary.json"
MANIFEST_FILE = "xunce-guarded-training-candidate-preflight-manifest.json"
SOURCE_AUDIT_FILE = "xunce-guarded-training-source-evidence-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-guarded-training-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-guarded-training-rejection-report.json"
REPORT_FILE = "xunce-guarded-training-candidate-preflight-report.md"

ARCHITECTURE = "xunce_full_network_v1"
PASS_NEXT_REQUIRED_CHANGE = "controlled_training_candidate"
FAIL_NEXT_REQUIRED_CHANGE = "fix_guarded_training_candidate_preflight"
FIX_STRESS_NEXT_REQUIRED_CHANGE = "fix_full_network_stress_evaluation"

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

SOURCE_FILES = {
    "full_network": ("source_full_network_root", "xunce-full-network-v1-summary.json", "full_network_v1_passed", "full_network_static_contract_validation"),
    "static_contract": ("source_static_contract_root", "xunce-full-network-static-contract-validation-summary.json", "static_contract_validation_passed", "full_network_ablation_experiments"),
    "ablation": ("source_ablation_root", "xunce-full-network-ablation-experiments-summary.json", "ablation_experiments_passed", "full_network_stress_evaluation"),
    "stress": ("source_stress_root", "xunce-full-network-stress-evaluation-summary.json", "stress_evaluation_passed", "guarded_training_candidate_preflight"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce Guarded Training Candidate Preflight v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_guarded_training_candidate_preflight(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"], "next_required_change": summary["next_required_change"], "summary": summary["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_guarded_training_candidate_preflight(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    sources = _load_sources(config, repo_root)
    source_audit = _source_audit(sources)
    boundary_audit = _boundary_audit(sources)
    decision = _decision(source_audit, boundary_audit)
    generated_at = utc_now()
    summary = _summary(generated_at, config_path, output_root, paths, sources, source_audit, boundary_audit, decision, repo_root)
    manifest = {"schema_version": MANIFEST_SCHEMA_VERSION, "generated_at": generated_at, "config": str(config_path), "output_root": str(output_root), "artifacts": {key: str(path) for key, path in paths.items()}, "summary_status": summary["status"], "next_required_change": summary["next_required_change"]}
    rejection_report = {"schema_version": REJECTION_REPORT_SCHEMA_VERSION, "status": decision["status"], "reason_codes": decision["reason_codes"], "next_required_change": decision["next_required_change"], "source_evidence_audit_passed": source_audit["source_evidence_audit_passed"], "boundary_audit_passed": boundary_audit["boundary_audit_passed"]}
    write_json(paths["source_audit"], source_audit)
    write_json(paths["boundary_audit"], boundary_audit)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, rejection_report), encoding="utf-8")
    return summary


def _load_config(path: Path) -> dict[str, Any]:
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
    if payload.get("architecture") != ARCHITECTURE:
        raise ConfigError(f"architecture must be {ARCHITECTURE!r}")
    for key, *_ in SOURCE_FILES.values():
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    return dict(payload)


def _load_sources(config: dict[str, Any], repo_root: Path) -> dict[str, dict[str, Any] | None]:
    sources = {}
    for name, (root_key, filename, _pass_field, _next) in SOURCE_FILES.items():
        sources[name] = _load_json(resolve_path(Path(config[root_key]), repo_root) / filename)
    return sources


def _source_audit(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    reason_codes: list[str] = []
    details = {}
    for name, (_root_key, _filename, pass_field, expected_next) in SOURCE_FILES.items():
        payload = sources.get(name)
        reason_label = "full_network_stress" if name == "stress" else name
        details[name] = {
            "status": payload.get("status") if isinstance(payload, dict) else None,
            "next_required_change": payload.get("next_required_change") if isinstance(payload, dict) else None,
            "pass_field": payload.get(pass_field) if isinstance(payload, dict) else None,
        }
        if not isinstance(payload, dict):
            reason_codes.append(f"missing_{reason_label}_summary")
            continue
        if payload.get("status") != "passed" or payload.get(pass_field) is not True:
            reason_codes.append(f"{reason_label}_not_passed")
        if payload.get("next_required_change") != expected_next:
            reason_codes.append(f"{reason_label}_wrong_next_required_change")
    return {
        "schema_version": SOURCE_AUDIT_SCHEMA_VERSION,
        "sources": details,
        "reason_codes": unique_sorted(reason_codes),
        "source_evidence_audit_passed": not reason_codes,
    }


def _boundary_audit(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    violations: list[str] = []
    observed: dict[str, dict[str, bool]] = {}
    for name, payload in sources.items():
        observed[name] = {}
        if not isinstance(payload, dict):
            continue
        for field in BOUNDARY_FIELDS:
            value = bool(payload.get(field, False))
            observed[name][field] = value
            if value:
                violations.append(f"{name}.{field}")
    return {"schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION, "observed_source_boundary_fields": observed, "violating_source_boundary_fields": unique_sorted(violations), "boundary_audit_passed": not violations, **_closed_boundary_fields()}


def _decision(source_audit: dict[str, Any], boundary_audit: dict[str, Any]) -> dict[str, Any]:
    reason_codes = list(source_audit["reason_codes"])
    if not boundary_audit["boundary_audit_passed"]:
        reason_codes.append("training_preflight_boundary_violation")
    reason_codes = unique_sorted(reason_codes)
    passed = not reason_codes
    if passed:
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif any(code.startswith("missing_full_network_stress") or code.startswith("full_network_stress") for code in reason_codes):
        next_required_change = FIX_STRESS_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FAIL_NEXT_REQUIRED_CHANGE
    return {"status": "passed" if passed else "failed", "reason_codes": reason_codes, "next_required_change": next_required_change}


def _summary(generated_at: str, config_path: Path, output_root: Path, paths: dict[str, Path], sources: dict[str, dict[str, Any] | None], source_audit: dict[str, Any], boundary_audit: dict[str, Any], decision: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_full_network_status": _status(sources, "full_network"),
        "source_static_contract_status": _status(sources, "static_contract"),
        "source_ablation_status": _status(sources, "ablation"),
        "source_stress_status": _status(sources, "stress"),
        "architecture": ARCHITECTURE,
        "full_network_v1_passed": _flag(sources, "full_network", "full_network_v1_passed"),
        "static_contract_validation_passed": _flag(sources, "static_contract", "static_contract_validation_passed"),
        "ablation_experiments_passed": _flag(sources, "ablation", "ablation_experiments_passed"),
        "stress_evaluation_passed": _flag(sources, "stress", "stress_evaluation_passed"),
        "training_candidate_preflight_passed": decision["status"] == "passed",
        "training_preflight_verdict": "eligible_for_controlled_training_candidate" if decision["status"] == "passed" else "blocked",
        "controlled_training_candidate_authorized": decision["status"] == "passed",
        "source_evidence_audit_passed": source_audit["source_evidence_audit_passed"],
        "boundary_audit_passed": boundary_audit["boundary_audit_passed"],
        "next_required_change": decision["next_required_change"],
        "git_provenance": {"current": git_snapshot(repo_root)},
    }
    summary.update(_closed_boundary_fields())
    return summary


def _status(sources: dict[str, dict[str, Any] | None], name: str) -> str | None:
    payload = sources.get(name)
    return payload.get("status") if isinstance(payload, dict) else None


def _flag(sources: dict[str, dict[str, Any] | None], name: str, field: str) -> bool:
    payload = sources.get(name)
    return bool(payload.get(field, False)) if isinstance(payload, dict) else False


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {"summary": output_root / SUMMARY_FILE, "manifest": output_root / MANIFEST_FILE, "source_audit": output_root / SOURCE_AUDIT_FILE, "boundary_audit": output_root / BOUNDARY_AUDIT_FILE, "rejection_report": output_root / REJECTION_REPORT_FILE, "report": output_root / REPORT_FILE}


def _closed_boundary_fields() -> dict[str, bool | float]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields.update({"real_world_release_approved": False, "real_world_performance_claimed": False, "default_policy_replacement_approved": False, "real_executor_connection_approved": False, "starts_online_canary": False, "canary_traffic_fraction": 0.0, "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False, "runs_new_ppo_update": False, "modifies_network": False, "modifies_action_space": False, "modifies_default_astar": False})
    return fields


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    return "\n".join(["# Xunce Guarded Training Candidate Preflight v1", "", f"- status: `{summary['status']}`", f"- reason_codes: `{summary['reason_codes']}`", f"- training_preflight_verdict: `{summary['training_preflight_verdict']}`", f"- next_required_change: `{summary['next_required_change']}`", "", "This is an authorization preflight only. It does not run PPO or write checkpoints.", ""])


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())
