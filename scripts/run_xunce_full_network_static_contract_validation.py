from __future__ import annotations

import argparse
import json
import sys
from math import isfinite
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from global_99_governance_common import global_99_boundary_defaults
    from xunce_full_network_common import XunceFullNetworkV1
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from scripts.global_99_governance_common import global_99_boundary_defaults
    from scripts.xunce_full_network_common import XunceFullNetworkV1


CONFIG_SCHEMA_VERSION = "xunce-full-network-static-contract-validation-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-full-network-static-contract-validation-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-full-network-static-contract-validation-manifest/v1"
CASE_ROW_SCHEMA_VERSION = "xunce-full-network-static-contract-case-row/v1"
AUDIT_SCHEMA_PREFIX = "xunce-full-network-static-contract"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-full-network-static-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-full-network-static-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_full_network_static_contract_validation_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_full_network_static_contract_validation_v1"

SUMMARY_FILE = "xunce-full-network-static-contract-validation-summary.json"
MANIFEST_FILE = "xunce-full-network-static-contract-validation-manifest.json"
CASES_FILE = "xunce-full-network-static-contract-cases.jsonl"
SHAPE_AUDIT_FILE = "xunce-full-network-static-shape-audit.json"
MASK_AUDIT_FILE = "xunce-full-network-static-mask-audit.json"
METADATA_AUDIT_FILE = "xunce-full-network-static-metadata-audit.json"
COMPATIBILITY_AUDIT_FILE = "xunce-full-network-static-compatibility-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-full-network-static-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-full-network-static-rejection-report.json"
REPORT_FILE = "xunce-full-network-static-contract-validation-report.md"

ARCHITECTURE = "xunce_full_network_v1"
PASS_NEXT_REQUIRED_CHANGE = "full_network_ablation_experiments"
FAIL_NEXT_REQUIRED_CHANGE = "fix_full_network_static_contract_validation"
FIX_STAGE8_NEXT_REQUIRED_CHANGE = "fix_xunce_full_network_v1"

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
    parser = argparse.ArgumentParser(description="Run Xunce Full Network Static Contract Validation v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_full_network_static_contract_validation(
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


def run_xunce_full_network_static_contract_validation(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    stage8_root = resolve_path(Path(config["source_full_network_root"]), repo_root)
    stage8_summary_path = stage8_root / "xunce-full-network-v1-summary.json"
    stage8_summary = _load_json(stage8_summary_path)
    boundary_audit = _boundary_audit(stage8_summary)
    source_reason_codes = _source_reason_codes(stage8_summary)

    case_rows: list[dict[str, Any]] = []
    shape_audit = _empty_audit("shape")
    mask_audit = _empty_audit("mask")
    metadata_audit = _empty_audit("metadata")
    compatibility_audit = _empty_audit("compatibility")
    if not source_reason_codes:
        case_rows, shape_audit, mask_audit, metadata_audit, compatibility_audit = _run_static_cases(config)

    decision = _decision(source_reason_codes, shape_audit, mask_audit, metadata_audit, compatibility_audit, boundary_audit)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        stage8_summary_path=stage8_summary_path,
        stage8_summary=stage8_summary,
        case_rows=case_rows,
        shape_audit=shape_audit,
        mask_audit=mask_audit,
        metadata_audit=metadata_audit,
        compatibility_audit=compatibility_audit,
        boundary_audit=boundary_audit,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)
    rejection_report = _rejection_report(decision, shape_audit, mask_audit, metadata_audit, compatibility_audit, boundary_audit)

    write_jsonl(paths["cases"], case_rows)
    write_json(paths["shape_audit"], shape_audit)
    write_json(paths["mask_audit"], mask_audit)
    write_json(paths["metadata_audit"], metadata_audit)
    write_json(paths["compatibility_audit"], compatibility_audit)
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
    normalized = dict(payload)
    if not isinstance(payload.get("source_full_network_root"), str) or not payload["source_full_network_root"].strip():
        raise ConfigError("source_full_network_root must be a non-empty string")
    for key in (
        "candidate_count",
        "edge_count",
        "candidate_feature_count",
        "edge_feature_count",
        "memory_feature_count",
        "context_feature_count",
        "hidden_dim",
        "message_passing_layers",
        "seed",
    ):
        normalized[key] = _positive_int(payload.get(key), key)
    normalized["missing_indicator_count"] = _nonnegative_int(payload.get("missing_indicator_count"), "missing_indicator_count")
    normalized["dropout"] = _nonnegative_float(payload.get("dropout"), "dropout")
    if normalized["dropout"] >= 1.0:
        raise ConfigError("dropout must be < 1.0")
    mask = payload.get("mask_case_action_mask")
    if not isinstance(mask, list) or len(mask) != normalized["candidate_count"] or any(not isinstance(value, bool) for value in mask):
        raise ConfigError("mask_case_action_mask must be a boolean list matching candidate_count")
    if not any(mask):
        raise ConfigError("mask_case_action_mask must contain at least one true value")
    normalized["mask_case_action_mask"] = mask
    return normalized


def _run_static_cases(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    torch.manual_seed(config["seed"])
    network = XunceFullNetworkV1(
        candidate_feature_count=config["candidate_feature_count"],
        edge_feature_count=config["edge_feature_count"],
        memory_feature_count=config["memory_feature_count"],
        context_feature_count=config["context_feature_count"],
        missing_indicator_count=config["missing_indicator_count"],
        hidden_dim=config["hidden_dim"],
        message_passing_layers=config["message_passing_layers"],
        dropout=config["dropout"],
    )
    network.eval()
    base = _base_tensors(config)
    cases = [
        ("full_topology", base, True),
        ("mask", {**base, "action_mask": torch.tensor([config["mask_case_action_mask"]], dtype=torch.bool)}, True),
        ("missing_indicator_default", {**base, "candidate_missing_indicators": None}, True),
        ("legacy_additive_compatibility", _legacy_tensors(config), True),
        ("metadata", base, True),
    ]
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for case_id, tensors, expected in cases:
            try:
                output = network(**tensors)
                row = _case_row(case_id, output, tensors["action_mask"], passed=expected)
            except Exception as exc:  # pragma: no cover - captured in artifacts
                row = {
                    "schema_version": CASE_ROW_SCHEMA_VERSION,
                    "case_id": case_id,
                    "passed": False,
                    "error": str(exc),
                }
            rows.append(row)
    full_case = _row(rows, "full_topology")
    mask_case = _row(rows, "mask")
    missing_case = _row(rows, "missing_indicator_default")
    legacy_case = _row(rows, "legacy_additive_compatibility")
    metadata = network.metadata()
    shape_audit = {
        "schema_version": f"{AUDIT_SCHEMA_PREFIX}-shape-audit/v1",
        "logits_shape": full_case.get("logits_shape"),
        "masked_logits_shape": full_case.get("masked_logits_shape"),
        "action_probs_shape": full_case.get("action_probs_shape"),
        "value_shape": full_case.get("value_shape"),
        "shape_contract_passed": full_case.get("passed", False)
        and full_case.get("logits_shape") == [1, config["candidate_count"]]
        and full_case.get("value_shape") == [1],
    }
    mask_audit = {
        "schema_version": f"{AUDIT_SCHEMA_PREFIX}-mask-audit/v1",
        "invalid_action_count": config["mask_case_action_mask"].count(False),
        "invalid_action_probability_sum": mask_case.get("invalid_action_probability_sum", 1.0),
        "valid_action_probability_sum": mask_case.get("valid_action_probability_sum", 0.0),
        "mask_contract_passed": config["mask_case_action_mask"].count(False) > 0
        and mask_case.get("passed", False)
        and mask_case.get("invalid_action_probability_sum", 1.0) == 0.0
        and abs(mask_case.get("valid_action_probability_sum", 0.0) - 1.0) < 1.0e-5,
    }
    metadata_audit = {
        "schema_version": f"{AUDIT_SCHEMA_PREFIX}-metadata-audit/v1",
        "metadata": metadata,
        "metadata_contract_passed": metadata.get("architecture") == ARCHITECTURE
        and metadata.get("candidate_feature_count") == config["candidate_feature_count"]
        and metadata.get("edge_feature_count") == config["edge_feature_count"]
        and metadata.get("memory_feature_count") == config["memory_feature_count"]
        and metadata.get("context_feature_count") == config["context_feature_count"]
        and metadata.get("candidate_graph_encoder_used") is True
        and metadata.get("topology_bias_used") is True
        and metadata.get("coverage_memory_token_used") is True
        and metadata.get("roi_budget_fusion_used") is True
        and metadata.get("production_registered") is False,
    }
    compatibility_audit = {
        "schema_version": f"{AUDIT_SCHEMA_PREFIX}-compatibility-audit/v1",
        "missing_indicator_contract_passed": missing_case.get("passed", False) and missing_case.get("non_finite_output_count") == 0,
        "legacy_observation_compatibility_passed": legacy_case.get("passed", False) and legacy_case.get("non_finite_output_count") == 0,
        "finite_output_contract_passed": all(row.get("passed", False) and row.get("non_finite_output_count", 1) == 0 for row in rows if row.get("case_id") != "metadata"),
    }
    return rows, shape_audit, mask_audit, metadata_audit, compatibility_audit


def _base_tensors(config: dict[str, Any]) -> dict[str, Any]:
    candidate_count = config["candidate_count"]
    edge_count = config["edge_count"]
    left = torch.arange(edge_count, dtype=torch.long) % candidate_count
    right = (left + 1).remainder(candidate_count)
    return {
        "candidate_features": torch.randn(1, candidate_count, config["candidate_feature_count"]),
        "edge_features": torch.randn(edge_count, config["edge_feature_count"]),
        "edge_index": torch.stack((left, right), dim=1),
        "memory_features": torch.randn(1, config["memory_feature_count"]),
        "context_features": torch.randn(1, candidate_count, config["context_feature_count"]),
        "action_mask": torch.ones(1, candidate_count, dtype=torch.bool),
        "candidate_missing_indicators": torch.zeros(1, candidate_count, config["missing_indicator_count"]),
    }


def _legacy_tensors(config: dict[str, Any]) -> dict[str, Any]:
    candidate_count = config["candidate_count"]
    return {
        "candidate_features": torch.randn(1, candidate_count, config["candidate_feature_count"]),
        "edge_features": torch.zeros(0, config["edge_feature_count"]),
        "edge_index": torch.zeros(0, 2, dtype=torch.long),
        "memory_features": torch.zeros(1, config["memory_feature_count"]),
        "context_features": torch.zeros(1, candidate_count, config["context_feature_count"]),
        "action_mask": torch.ones(1, candidate_count, dtype=torch.bool),
        "candidate_missing_indicators": None,
    }


def _case_row(case_id: str, output, action_mask: torch.Tensor, *, passed: bool) -> dict[str, Any]:
    logits = output.masked_logits[0].detach().cpu().tolist()
    probs = output.action_probs[0].detach().cpu().tolist()
    mask = action_mask[0].detach().cpu().tolist()
    non_finite = sum(0 if isfinite(float(value)) else 1 for value in logits + probs + [float(output.value[0].detach().cpu())])
    return {
        "schema_version": CASE_ROW_SCHEMA_VERSION,
        "case_id": case_id,
        "passed": passed and non_finite == 0,
        "logits_shape": list(output.logits.shape),
        "masked_logits_shape": list(output.masked_logits.shape),
        "action_probs_shape": list(output.action_probs.shape),
        "value_shape": list(output.value.shape),
        "valid_action_probability_sum": sum(float(prob) for prob, valid in zip(probs, mask) if valid),
        "invalid_action_probability_sum": sum(float(prob) for prob, valid in zip(probs, mask) if not valid),
        "non_finite_output_count": non_finite,
    }


def _row(rows: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    for row in rows:
        if row.get("case_id") == case_id:
            return row
    return {}


def _decision(
    source_reason_codes: list[str],
    shape_audit: dict[str, Any],
    mask_audit: dict[str, Any],
    metadata_audit: dict[str, Any],
    compatibility_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    reason_codes = list(source_reason_codes)
    source_blocked = bool(source_reason_codes)
    next_required_change = FIX_STAGE8_NEXT_REQUIRED_CHANGE if source_blocked else FAIL_NEXT_REQUIRED_CHANGE
    if not boundary_audit.get("boundary_audit_passed", False):
        reason_codes.append("static_contract_boundary_violation")
    if not source_blocked:
        if not shape_audit.get("shape_contract_passed", False):
            reason_codes.append("static_shape_contract_failed")
        if not mask_audit.get("mask_contract_passed", False):
            reason_codes.append("static_mask_contract_failed")
        if not metadata_audit.get("metadata_contract_passed", False):
            reason_codes.append("static_metadata_contract_failed")
        if not compatibility_audit.get("missing_indicator_contract_passed", False):
            reason_codes.append("static_missing_indicator_contract_failed")
        if not compatibility_audit.get("legacy_observation_compatibility_passed", False):
            reason_codes.append("static_legacy_observation_compatibility_failed")
        if not compatibility_audit.get("finite_output_contract_passed", False):
            reason_codes.append("static_finite_output_contract_failed")
    reason_codes = unique_sorted(reason_codes)
    passed = not reason_codes
    return {
        "status": "passed" if passed else "failed",
        "reason_codes": reason_codes,
        "next_required_change": PASS_NEXT_REQUIRED_CHANGE if passed else next_required_change,
    }


def _source_reason_codes(stage8_summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(stage8_summary, dict):
        return ["missing_full_network_v1_summary"]
    reason_codes: list[str] = []
    if stage8_summary.get("status") != "passed":
        reason_codes.append("full_network_v1_not_passed")
    if stage8_summary.get("next_required_change") != "full_network_static_contract_validation":
        reason_codes.append("full_network_v1_wrong_next_required_change")
    return reason_codes


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    stage8_summary_path: Path,
    stage8_summary: dict[str, Any] | None,
    case_rows: list[dict[str, Any]],
    shape_audit: dict[str, Any],
    mask_audit: dict[str, Any],
    metadata_audit: dict[str, Any],
    compatibility_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_full_network_summary": str(stage8_summary_path),
        "source_full_network_status": stage8_summary.get("status") if isinstance(stage8_summary, dict) else None,
        "architecture": ARCHITECTURE,
        "contract_case_count": len(case_rows),
        "shape_contract_passed": shape_audit.get("shape_contract_passed", False),
        "mask_contract_passed": mask_audit.get("mask_contract_passed", False),
        "metadata_contract_passed": metadata_audit.get("metadata_contract_passed", False),
        "missing_indicator_contract_passed": compatibility_audit.get("missing_indicator_contract_passed", False),
        "legacy_observation_compatibility_passed": compatibility_audit.get("legacy_observation_compatibility_passed", False),
        "finite_output_contract_passed": compatibility_audit.get("finite_output_contract_passed", False),
        "boundary_audit_passed": boundary_audit.get("boundary_audit_passed", False),
        "static_contract_validation_passed": decision["status"] == "passed",
        "next_required_change": decision["next_required_change"],
        "git_provenance": {"current": git_snapshot(repo_root)},
    }
    summary.update(_closed_boundary_fields())
    return summary


def _manifest(generated_at: str, config_path: Path, output_root: Path, paths: dict[str, Path], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(path) for key, path in paths.items()},
        "summary_status": summary["status"],
        "next_required_change": summary["next_required_change"],
    }


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "cases": output_root / CASES_FILE,
        "shape_audit": output_root / SHAPE_AUDIT_FILE,
        "mask_audit": output_root / MASK_AUDIT_FILE,
        "metadata_audit": output_root / METADATA_AUDIT_FILE,
        "compatibility_audit": output_root / COMPATIBILITY_AUDIT_FILE,
        "boundary_audit": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _boundary_audit(stage8_summary: dict[str, Any] | None) -> dict[str, Any]:
    observed = {}
    violations = []
    if isinstance(stage8_summary, dict):
        for field in BOUNDARY_FIELDS:
            value = bool(stage8_summary.get(field, False))
            observed[field] = value
            if value:
                violations.append(field)
    return {
        "schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION,
        "observed_source_boundary_fields": observed,
        "violating_source_boundary_fields": sorted(set(violations)),
        "boundary_audit_passed": not violations,
        **_closed_boundary_fields(),
    }


def _closed_boundary_fields() -> dict[str, bool | float]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields.update(
        {
            "real_world_release_approved": False,
            "real_world_performance_claimed": False,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
        }
    )
    return fields


def _rejection_report(
    decision: dict[str, Any],
    shape_audit: dict[str, Any],
    mask_audit: dict[str, Any],
    metadata_audit: dict[str, Any],
    compatibility_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "shape_contract_passed": shape_audit.get("shape_contract_passed", False),
        "mask_contract_passed": mask_audit.get("mask_contract_passed", False),
        "metadata_contract_passed": metadata_audit.get("metadata_contract_passed", False),
        "missing_indicator_contract_passed": compatibility_audit.get("missing_indicator_contract_passed", False),
        "legacy_observation_compatibility_passed": compatibility_audit.get("legacy_observation_compatibility_passed", False),
        "boundary_audit_passed": boundary_audit.get("boundary_audit_passed", False),
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    lines = [
        "# Xunce Full Network Static Contract Validation v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- static_contract_validation_passed: `{summary['static_contract_validation_passed']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        "",
        "## Contracts",
        "",
        f"- shape_contract_passed: `{rejection_report['shape_contract_passed']}`",
        f"- mask_contract_passed: `{rejection_report['mask_contract_passed']}`",
        f"- metadata_contract_passed: `{rejection_report['metadata_contract_passed']}`",
        f"- missing_indicator_contract_passed: `{rejection_report['missing_indicator_contract_passed']}`",
        f"- legacy_observation_compatibility_passed: `{rejection_report['legacy_observation_compatibility_passed']}`",
        "",
        "This is a static contract gate only. It does not train, publish, install, or connect an executor.",
        "",
    ]
    return "\n".join(lines)


def _empty_audit(name: str) -> dict[str, Any]:
    return {
        "schema_version": f"{AUDIT_SCHEMA_PREFIX}-{name}-audit/v1",
        f"{name}_contract_passed": False,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a positive integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a positive integer") from exc
    if numeric <= 0:
        raise ConfigError(f"{field_name} must be a positive integer")
    return numeric


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a non-negative integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a non-negative integer") from exc
    if numeric < 0:
        raise ConfigError(f"{field_name} must be a non-negative integer")
    return numeric


def _nonnegative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be non-negative")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be non-negative") from exc
    if numeric < 0.0:
        raise ConfigError(f"{field_name} must be non-negative")
    return numeric


if __name__ == "__main__":
    raise SystemExit(main())
