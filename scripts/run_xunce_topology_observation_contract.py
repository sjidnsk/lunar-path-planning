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


CONFIG_SCHEMA_VERSION = "xunce-topology-observation-contract-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-topology-observation-contract-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-topology-observation-contract-manifest/v1"
CONTRACT_SCHEMA_VERSION = "xunce-topology-observation-contract/v1"
CONTRACT_AUDIT_SCHEMA_VERSION = "xunce-topology-observation-contract-audit/v1"
COMPATIBILITY_AUDIT_SCHEMA_VERSION = "xunce-topology-observation-compatibility-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-topology-observation-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-topology-observation-contract-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_topology_observation_contract_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_topology_observation_contract_v1"

SUMMARY_FILE = "xunce-topology-observation-contract-summary.json"
MANIFEST_FILE = "xunce-topology-observation-contract-manifest.json"
CONTRACT_FILE = "xunce-topology-observation-contract.json"
CONTRACT_AUDIT_FILE = "xunce-topology-observation-contract-audit.json"
COMPATIBILITY_AUDIT_FILE = "xunce-topology-observation-compatibility-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-topology-observation-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-topology-observation-contract-rejection-report.json"
REPORT_FILE = "xunce-topology-observation-contract-report.md"

PASS_NEXT_REQUIRED_CHANGE = "topology_feature_extraction_audit"
FIX_STAGE2_NEXT_REQUIRED_CHANGE = "fix_xunce_network_literature_bottleneck_review"
FIX_CONTRACT_NEXT_REQUIRED_CHANGE = "fix_xunce_topology_observation_contract"

EXPECTED_BASE_OBSERVATION_SCHEMA = "policy-observation/v1.1"
EXPECTED_EXTENSION_SCHEMA = "xunce-topology-observation-extension/v1"
ALLOWED_FIELD_TYPES = {"numeric", "id", "bool"}
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce Topology Observation Contract v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_topology_observation_contract(
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
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_topology_observation_contract(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(Path(config_path), repo_root)
    paths = _artifact_paths(output_root)
    stage2_path = resolve_path(Path(config["source_network_literature_bottleneck_review_root"]), repo_root) / (
        "xunce-network-literature-bottleneck-review-summary.json"
    )
    stage2 = _load_json(stage2_path)
    contract = _contract_payload(config)
    contract_audit = _contract_audit(config, contract)
    compatibility_audit = _compatibility_audit(config)
    boundary_audit = _boundary_audit(stage2)
    decision = _decision(stage2, contract_audit, compatibility_audit, boundary_audit)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=Path(config_path),
        output_root=output_root,
        paths=paths,
        stage2_path=stage2_path,
        contract=contract,
        contract_audit=contract_audit,
        compatibility_audit=compatibility_audit,
        boundary_audit=boundary_audit,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, Path(config_path), output_root, paths, summary)
    rejection_report = _rejection_report(generated_at, decision, contract_audit, compatibility_audit, boundary_audit)

    write_json(paths["contract"], contract)
    write_json(paths["contract_audit"], contract_audit)
    write_json(paths["compatibility_audit"], compatibility_audit)
    write_json(paths["boundary_audit"], boundary_audit)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, contract, rejection_report), encoding="utf-8")
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
    normalized = dict(payload)
    for key in (
        "source_network_literature_bottleneck_review_root",
        "features_path",
        "torch_policy_path",
        "training_path",
        "architectures_path",
    ):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(payload[key]), repo_root))
    for key in ("candidate_topology_fields", "candidate_edge_fields", "coverage_memory_fields"):
        if not isinstance(payload.get(key), list) or not payload[key]:
            raise ConfigError(f"{key} must be a non-empty list")
        normalized[key] = [dict(item) for item in payload[key] if isinstance(item, dict)]
    if not isinstance(payload.get("contract"), dict):
        raise ConfigError("contract must be an object")
    return normalized


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "contract": output_root / CONTRACT_FILE,
        "contract_audit": output_root / CONTRACT_AUDIT_FILE,
        "compatibility_audit": output_root / COMPATIBILITY_AUDIT_FILE,
        "boundary_audit": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "base_observation_schema_version": config.get("base_observation_schema_version"),
        "extension_schema_version": config.get("extension_schema_version"),
        "contract": config["contract"],
        "candidate_topology_fields": config["candidate_topology_fields"],
        "candidate_edge_fields": config["candidate_edge_fields"],
        "coverage_memory_fields": config["coverage_memory_fields"],
    }


def _contract_audit(config: dict[str, Any], contract_payload: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    contract = config["contract"]
    if config.get("base_observation_schema_version") != EXPECTED_BASE_OBSERVATION_SCHEMA:
        reasons.append("base_observation_schema_changed")
    if config.get("extension_schema_version") != EXPECTED_EXTENSION_SCHEMA:
        reasons.append("extension_schema_invalid")
    if contract.get("additive_only") is not True:
        reasons.append("topology_contract_not_additive")
    for key in ("preserve_action_mask", "preserve_candidate_order", "optional_by_default", "require_missing_indicators"):
        if contract.get(key) is not True:
            reasons.append(f"topology_contract_{key}_not_true")
    field_groups = {
        "candidate_topology_fields": config["candidate_topology_fields"],
        "candidate_edge_fields": config["candidate_edge_fields"],
        "coverage_memory_fields": config["coverage_memory_fields"],
    }
    all_fields = [field for fields in field_groups.values() for field in fields]
    duplicate_names = _duplicate_names(all_fields)
    invalid_fields = []
    missing_indicator_fields = []
    required_fields = []
    for field in all_fields:
        name = str(field.get("name", "")).strip()
        field_type = str(field.get("type", "")).strip()
        if not name or field_type not in ALLOWED_FIELD_TYPES:
            invalid_fields.append(name or "<missing-name>")
        if field.get("optional") is not True:
            required_fields.append(name or "<missing-name>")
        if not str(field.get("missing_indicator", "")).strip():
            missing_indicator_fields.append(name or "<missing-name>")
    if duplicate_names:
        reasons.append("topology_contract_duplicate_fields")
    if invalid_fields:
        reasons.append("topology_contract_invalid_fields")
    if required_fields:
        reasons.append("topology_contract_fields_not_optional")
    if missing_indicator_fields:
        reasons.append("topology_contract_missing_indicators")
    return {
        "schema_version": CONTRACT_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed",
        "reason_codes": unique_sorted(reasons),
        "base_observation_schema_version": contract_payload["base_observation_schema_version"],
        "extension_schema_version": contract_payload["extension_schema_version"],
        "additive_only": contract.get("additive_only") is True,
        "preserve_action_mask": contract.get("preserve_action_mask") is True,
        "preserve_candidate_order": contract.get("preserve_candidate_order") is True,
        "all_new_fields_optional": not required_fields,
        "all_new_fields_have_missing_indicators": not missing_indicator_fields,
        "candidate_topology_field_count": len(config["candidate_topology_fields"]),
        "candidate_edge_field_count": len(config["candidate_edge_fields"]),
        "coverage_memory_field_count": len(config["coverage_memory_fields"]),
        "duplicate_field_names": duplicate_names,
        "invalid_fields": invalid_fields,
        "required_fields": required_fields,
        "missing_indicator_fields": missing_indicator_fields,
    }


def _compatibility_audit(config: dict[str, Any]) -> dict[str, Any]:
    source_text = {}
    for key in ("features_path", "torch_policy_path", "training_path", "architectures_path"):
        path = Path(config[key])
        source_text[key] = path.read_text(encoding="utf-8") if path.is_file() else ""
    checks = {
        "base_schema_present": EXPECTED_BASE_OBSERVATION_SCHEMA in source_text["features_path"],
        "action_mask_present": "action_mask" in source_text["features_path"] and "action_mask" in source_text["torch_policy_path"],
        "missing_indicators_present": "candidate_missing_indicator" in source_text["features_path"]
        and "candidate_missing_indicator" in source_text["training_path"],
        "torch_policy_scorer_present": "TorchPolicyScorer" in source_text["torch_policy_path"],
        "legacy_checkpoint_keys_present": "load_policy_checkpoint" in source_text["training_path"]
        and "checkpoint.get" in source_text["training_path"]
        and "state_dict" in source_text["training_path"],
        "current_architectures_present": all(
            marker in source_text["architectures_path"]
            for marker in ("mlp_v1", "mlp_missing_v1", "candidate_attention_v1")
        ),
    }
    reasons = [f"compatibility_{key}_missing" for key, value in checks.items() if not value]
    return {
        "schema_version": COMPATIBILITY_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed",
        "reason_codes": reasons,
        **checks,
    }


def _boundary_audit(stage2: dict[str, Any] | None) -> dict[str, Any]:
    violations = []
    if isinstance(stage2, dict):
        for field in BOUNDARY_FIELDS:
            if stage2.get(field) is True:
                violations.append({"source": "network_literature_bottleneck_review", "field": field, "value": True})
    return {
        "schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not violations else "failed",
        "reason_codes": [] if not violations else ["topology_observation_boundary_violation"],
        "violations": violations,
        **global_99_boundary_defaults(),
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
    }


def _decision(
    stage2: dict[str, Any] | None,
    contract_audit: dict[str, Any],
    compatibility_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(stage2, dict):
        reasons.append("missing_network_literature_bottleneck_review_summary")
    else:
        if stage2.get("status") != "passed":
            reasons.append("literature_bottleneck_review_not_passed")
        if stage2.get("next_required_change") != "topology_observation_contract":
            reasons.append("literature_bottleneck_review_next_required_change_invalid")
    reasons.extend(contract_audit.get("reason_codes", []))
    reasons.extend(compatibility_audit.get("reason_codes", []))
    reasons.extend(boundary_audit.get("reason_codes", []))
    reasons = unique_sorted(reasons)
    status = "passed" if not reasons else "failed"
    if status == "passed":
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif "literature_bottleneck_review_not_passed" in reasons or "literature_bottleneck_review_next_required_change_invalid" in reasons or "missing_network_literature_bottleneck_review_summary" in reasons:
        next_required_change = FIX_STAGE2_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FIX_CONTRACT_NEXT_REQUIRED_CHANGE
    return {"status": status, "reason_codes": reasons, "next_required_change": next_required_change}


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    stage2_path: Path,
    contract: dict[str, Any],
    contract_audit: dict[str, Any],
    compatibility_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "summary": str(paths["summary"]),
        "config_path": str(config_path),
        "output_root": str(output_root),
        "source_network_literature_bottleneck_review_summary": str(stage2_path),
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "topology_observation_contract_passed": decision["status"] == "passed",
        "contract_audit_passed": contract_audit["status"] == "passed",
        "compatibility_audit_passed": compatibility_audit["status"] == "passed",
        "boundary_audit_passed": boundary_audit["status"] == "passed",
        "base_observation_schema_version": contract["base_observation_schema_version"],
        "extension_schema_version": contract["extension_schema_version"],
        "candidate_topology_field_count": contract_audit["candidate_topology_field_count"],
        "candidate_edge_field_count": contract_audit["candidate_edge_field_count"],
        "coverage_memory_field_count": contract_audit["coverage_memory_field_count"],
        "additive_only": contract_audit["additive_only"],
        "all_new_fields_optional": contract_audit["all_new_fields_optional"],
        "all_new_fields_have_missing_indicators": contract_audit["all_new_fields_have_missing_indicators"],
        "next_required_change": decision["next_required_change"],
        **global_99_boundary_defaults(),
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "git_provenance": {"current": git_snapshot(repo_root)},
    }


def _manifest(generated_at: str, config_path: Path, output_root: Path, paths: dict[str, Path], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config_path": str(config_path),
        "output_root": str(output_root),
        "status": summary["status"],
        "next_required_change": summary["next_required_change"],
        "artifacts": {key: str(value) for key, value in paths.items()},
    }


def _rejection_report(
    generated_at: str,
    decision: dict[str, Any],
    contract_audit: dict[str, Any],
    compatibility_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed" if decision["status"] == "passed" else "failed",
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "contract_rejections": contract_audit.get("reason_codes", []),
        "compatibility_rejections": compatibility_audit.get("reason_codes", []),
        "boundary_rejections": boundary_audit.get("violations", []),
    }


def _render_report(summary: dict[str, Any], contract: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Topology Observation Contract v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- base_observation_schema_version: `{summary['base_observation_schema_version']}`",
            f"- candidate_topology_field_count: `{summary['candidate_topology_field_count']}`",
            f"- candidate_edge_field_count: `{summary['candidate_edge_field_count']}`",
            f"- coverage_memory_field_count: `{summary['coverage_memory_field_count']}`",
            "",
            "## Contract",
            "",
            json.dumps(contract, ensure_ascii=False, indent=2),
            "",
            "## Rejections",
            "",
            json.dumps(rejection_report, ensure_ascii=False, indent=2),
            "",
        ]
    )


def _duplicate_names(fields: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for field in fields:
        name = str(field.get("name", "")).strip()
        if not name:
            continue
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    return sorted(duplicates)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
