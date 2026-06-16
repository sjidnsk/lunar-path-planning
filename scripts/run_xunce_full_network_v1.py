from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
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
    from xunce_full_network_common import XunceFullNetworkV1, parameter_count
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from scripts.global_99_governance_common import global_99_boundary_defaults
    from scripts.xunce_full_network_common import XunceFullNetworkV1, parameter_count


CONFIG_SCHEMA_VERSION = "xunce-full-network-v1-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-full-network-v1-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-full-network-v1-manifest/v1"
FORWARD_AUDIT_SCHEMA_VERSION = "xunce-full-network-v1-forward-audit/v1"
METADATA_AUDIT_SCHEMA_VERSION = "xunce-full-network-v1-metadata-audit/v1"
PARAM_LATENCY_AUDIT_SCHEMA_VERSION = "xunce-full-network-v1-parameter-latency-audit/v1"
LOGIT_ROW_SCHEMA_VERSION = "xunce-full-network-v1-logit-row/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-full-network-v1-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-full-network-v1-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_full_network_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_full_network_v1"

SUMMARY_FILE = "xunce-full-network-v1-summary.json"
MANIFEST_FILE = "xunce-full-network-v1-manifest.json"
FORWARD_AUDIT_FILE = "xunce-full-network-v1-forward-audit.json"
METADATA_AUDIT_FILE = "xunce-full-network-v1-metadata-audit.json"
PARAM_LATENCY_AUDIT_FILE = "xunce-full-network-v1-parameter-latency-audit.json"
LOGITS_FILE = "xunce-full-network-v1-logits.jsonl"
BOUNDARY_AUDIT_FILE = "xunce-full-network-v1-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-full-network-v1-rejection-report.json"
REPORT_FILE = "xunce-full-network-v1-report.md"

ARCHITECTURE = "xunce_full_network_v1"
PASS_NEXT_REQUIRED_CHANGE = "full_network_static_contract_validation"
FAIL_NEXT_REQUIRED_CHANGE = "fix_xunce_full_network_v1"
FIX_STAGE7_NEXT_REQUIRED_CHANGE = "fix_architecture_contrast_evaluation"
FIX_STAGE4_NEXT_REQUIRED_CHANGE = "fix_xunce_topology_feature_extraction_audit"

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
    parser = argparse.ArgumentParser(description="Run Xunce Full Network v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_full_network_v1(
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
                "architecture": summary["architecture"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_full_network_v1(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config_path = Path(config_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    paths = _artifact_paths(output_root)
    stage7_root = resolve_path(Path(config["source_architecture_contrast_root"]), repo_root)
    stage4_root = resolve_path(Path(config["source_topology_feature_extraction_root"]), repo_root)
    stage7_summary_path = stage7_root / "xunce-architecture-contrast-evaluation-summary.json"
    stage7_summary = _load_json(stage7_summary_path)
    boundary_audit = _boundary_audit(stage7_summary)
    source_reason_codes = _source_reason_codes(stage7_summary)

    network_result = _empty_network_result()
    if not source_reason_codes:
        source_paths = _source_paths(stage4_root)
        candidates = _load_jsonl(source_paths["candidates"])
        edges = _load_jsonl(source_paths["edges"])
        memory = _load_json(source_paths["memory"])
        if not candidates:
            source_reason_codes.append("missing_topology_feature_candidates")
        if not edges:
            source_reason_codes.append("missing_topology_feature_edges")
        if not isinstance(memory, dict):
            source_reason_codes.append("missing_topology_feature_memory")
        if not source_reason_codes:
            network_result = _run_network(config, candidates, edges, memory)

    forward_audit = network_result["forward_audit"]
    metadata_audit = network_result["metadata_audit"]
    param_latency_audit = network_result["parameter_latency_audit"]
    decision = _decision(source_reason_codes, forward_audit, metadata_audit, param_latency_audit, boundary_audit)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        config=config,
        stage7_summary_path=stage7_summary_path,
        stage7_summary=stage7_summary,
        network_result=network_result,
        forward_audit=forward_audit,
        metadata_audit=metadata_audit,
        param_latency_audit=param_latency_audit,
        boundary_audit=boundary_audit,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)
    rejection_report = _rejection_report(decision, forward_audit, metadata_audit, param_latency_audit, boundary_audit)

    write_json(paths["forward_audit"], forward_audit)
    write_json(paths["metadata_audit"], metadata_audit)
    write_json(paths["param_latency_audit"], param_latency_audit)
    write_jsonl(paths["logits"], network_result["logit_rows"])
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
    for key in ("source_architecture_contrast_root", "source_topology_feature_extraction_root"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    for key in ("hidden_dim", "message_passing_layers", "seed", "latency_repeats", "max_parameter_count"):
        normalized[key] = _positive_int(payload.get(key), key)
    for key in ("dropout", "max_forward_latency_ms"):
        normalized[key] = _nonnegative_float(payload.get(key), key)
    if normalized["dropout"] >= 1.0:
        raise ConfigError("dropout must be < 1.0")
    mask = payload.get("action_mask_override")
    if mask is not None:
        if not isinstance(mask, list) or not mask or any(not isinstance(value, bool) for value in mask):
            raise ConfigError("action_mask_override must be a non-empty boolean list")
        if not any(mask):
            raise ConfigError("action_mask_override must contain at least one true value")
        normalized["action_mask_override"] = mask
    return normalized


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "forward_audit": output_root / FORWARD_AUDIT_FILE,
        "metadata_audit": output_root / METADATA_AUDIT_FILE,
        "param_latency_audit": output_root / PARAM_LATENCY_AUDIT_FILE,
        "logits": output_root / LOGITS_FILE,
        "boundary_audit": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _source_paths(stage4_root: Path) -> dict[str, Path]:
    return {
        "candidates": stage4_root / "xunce-topology-feature-extraction-candidates.jsonl",
        "edges": stage4_root / "xunce-topology-feature-extraction-edges.jsonl",
        "memory": stage4_root / "xunce-topology-feature-extraction-memory.json",
    }


def _run_network(config: dict[str, Any], candidates: list[dict[str, Any]], edges: list[dict[str, Any]], memory: dict[str, Any]) -> dict[str, Any]:
    tensors = _fixture_tensors(candidates, edges, memory, config.get("action_mask_override"))
    torch.manual_seed(config["seed"])
    network = XunceFullNetworkV1(
        candidate_feature_count=len(tensors["candidate_feature_names"]),
        edge_feature_count=len(tensors["edge_feature_names"]),
        memory_feature_count=len(tensors["memory_feature_names"]),
        context_feature_count=len(tensors["context_feature_names"]),
        missing_indicator_count=len(tensors["candidate_missing_indicator_names"]),
        hidden_dim=config["hidden_dim"],
        message_passing_layers=config["message_passing_layers"],
        dropout=config["dropout"],
    )
    network.eval()
    timings: list[float] = []
    output = None
    with torch.no_grad():
        for _ in range(config["latency_repeats"]):
            started = time.perf_counter()
            output = network(
                candidate_features=tensors["candidate_features"],
                edge_features=tensors["edge_features"],
                edge_index=tensors["edge_index"],
                memory_features=tensors["memory_features"],
                context_features=tensors["context_features"],
                action_mask=tensors["action_mask"],
                candidate_missing_indicators=tensors["candidate_missing_indicators"],
            )
            timings.append((time.perf_counter() - started) * 1000.0)
    if output is None:
        raise ConfigError("latency_repeats must be positive")
    metadata = network.metadata()
    param_count = parameter_count(network)
    median_latency = float(statistics.median(timings))
    logits = output.masked_logits[0].detach().cpu().tolist()
    probs = output.action_probs[0].detach().cpu().tolist()
    logit_rows = [
        {
            "schema_version": LOGIT_ROW_SCHEMA_VERSION,
            "candidate_index": index,
            "masked_logit": float(logits[index]),
            "action_probability": float(probs[index]),
            "action_mask": bool(tensors["action_mask"][0, index].item()),
        }
        for index in range(len(logits))
    ]
    non_finite_output_count = sum(0 if isfinite(float(value)) else 1 for value in logits + probs + [float(output.value[0].detach().cpu())])
    valid_probability_sum = sum(float(prob) for prob, mask in zip(probs, tensors["action_mask"][0].detach().cpu().tolist()) if mask)
    invalid_probability_sum = sum(float(prob) for prob, mask in zip(probs, tensors["action_mask"][0].detach().cpu().tolist()) if not mask)
    forward_audit = {
        "schema_version": FORWARD_AUDIT_SCHEMA_VERSION,
        "logits_shape": list(output.logits.shape),
        "masked_logits_shape": list(output.masked_logits.shape),
        "action_probs_shape": list(output.action_probs.shape),
        "value_shape": list(output.value.shape),
        "valid_action_probability_sum": valid_probability_sum,
        "invalid_action_probability_sum": invalid_probability_sum,
        "non_finite_output_count": non_finite_output_count,
        "masked_logits_valid": non_finite_output_count == 0 and abs(valid_probability_sum - 1.0) < 1.0e-5 and invalid_probability_sum == 0.0,
        "value_head_valid": isfinite(float(output.value[0].detach().cpu())),
    }
    metadata_audit = {
        "schema_version": METADATA_AUDIT_SCHEMA_VERSION,
        "metadata": metadata,
        "metadata_audit_passed": (
            metadata.get("architecture") == ARCHITECTURE
            and metadata.get("candidate_graph_encoder_used") is True
            and metadata.get("topology_bias_used") is True
            and metadata.get("coverage_memory_token_used") is True
            and metadata.get("roi_budget_fusion_used") is True
            and metadata.get("production_registered") is False
        ),
    }
    parameter_latency_audit = {
        "schema_version": PARAM_LATENCY_AUDIT_SCHEMA_VERSION,
        "parameter_count": param_count,
        "max_parameter_count": config["max_parameter_count"],
        "median_forward_latency_ms": median_latency,
        "max_forward_latency_ms": config["max_forward_latency_ms"],
        "parameter_latency_audit_passed": param_count <= config["max_parameter_count"] and median_latency <= config["max_forward_latency_ms"],
    }
    return {
        "candidate_count": len(candidates),
        "edge_count": len(edges),
        "metadata": metadata,
        "forward_audit": forward_audit,
        "metadata_audit": metadata_audit,
        "parameter_latency_audit": parameter_latency_audit,
        "logit_rows": logit_rows,
    }


def _fixture_tensors(
    candidates: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    memory: dict[str, Any],
    action_mask_override: list[bool] | None,
) -> dict[str, Any]:
    candidate_feature_names = sorted(candidates[0].get("features", {}))
    candidate_missing_indicator_names = sorted(candidates[0].get("missing_indicators", {}))
    edge_feature_names = sorted(edges[0].get("features", {}))
    memory_feature_names = sorted(memory.get("features", {}))
    context_feature_names = [
        name
        for name in candidate_feature_names
        if name in {"budget_fraction_cost", "fallback_risk", "frontier_cluster_id", "path_bottleneck_score", "revisit_path_cell_count", "roi_group_id"}
    ] or candidate_feature_names
    mask = action_mask_override or [True] * len(candidates)
    if len(mask) != len(candidates):
        raise ConfigError("action_mask_override length must match candidate count")
    return {
        "candidate_feature_names": candidate_feature_names,
        "candidate_missing_indicator_names": candidate_missing_indicator_names,
        "edge_feature_names": edge_feature_names,
        "memory_feature_names": memory_feature_names,
        "context_feature_names": context_feature_names,
        "candidate_features": torch.tensor(
            [[[_feature_value(candidate.get("features", {}), name) for name in candidate_feature_names] for candidate in candidates]],
            dtype=torch.float32,
        ),
        "candidate_missing_indicators": torch.tensor(
            [[[1.0 if candidate.get("missing_indicators", {}).get(name, False) else 0.0 for name in candidate_missing_indicator_names] for candidate in candidates]],
            dtype=torch.float32,
        ),
        "edge_features": torch.tensor(
            [[_feature_value(edge.get("features", {}), name) for name in edge_feature_names] for edge in edges],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor(
            [[int(edge["candidate_left_index"]), int(edge["candidate_right_index"])] for edge in edges],
            dtype=torch.long,
        ),
        "memory_features": torch.tensor(
            [[_feature_value(memory.get("features", {}), name) for name in memory_feature_names]],
            dtype=torch.float32,
        ),
        "context_features": torch.tensor(
            [[[_feature_value(candidate.get("features", {}), name) for name in context_feature_names] for candidate in candidates]],
            dtype=torch.float32,
        ),
        "action_mask": torch.tensor([mask], dtype=torch.bool),
    }


def _feature_value(features: dict[str, Any], name: str) -> float:
    value = features.get(name, 0.0)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _source_reason_codes(stage7_summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(stage7_summary, dict):
        return ["missing_architecture_contrast_evaluation_summary"]
    reason_codes: list[str] = []
    if stage7_summary.get("status") != "passed":
        reason_codes.append("architecture_contrast_evaluation_not_passed")
    if stage7_summary.get("next_required_change") != "full_xunce_network_v1_design":
        reason_codes.append("architecture_contrast_evaluation_wrong_next_required_change")
    return reason_codes


def _decision(
    source_reason_codes: list[str],
    forward_audit: dict[str, Any],
    metadata_audit: dict[str, Any],
    param_latency_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    reason_codes = list(source_reason_codes)
    source_blocked = bool(source_reason_codes)
    if source_blocked and any(code.startswith("missing_architecture") or code.startswith("architecture_contrast") for code in source_reason_codes):
        next_required_change = FIX_STAGE7_NEXT_REQUIRED_CHANGE
    elif source_blocked and any(code.startswith("missing_topology_feature") for code in source_reason_codes):
        next_required_change = FIX_STAGE4_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FAIL_NEXT_REQUIRED_CHANGE
    if not boundary_audit.get("boundary_audit_passed", False):
        reason_codes.append("full_network_boundary_violation")
    if not source_blocked:
        if not forward_audit.get("masked_logits_valid", False) or not forward_audit.get("value_head_valid", False):
            reason_codes.append("full_network_forward_audit_failed")
        if not metadata_audit.get("metadata_audit_passed", False):
            reason_codes.append("full_network_metadata_audit_failed")
        if not param_latency_audit.get("parameter_latency_audit_passed", False):
            reason_codes.append("full_network_parameter_latency_gate_failed")
    reason_codes = unique_sorted(reason_codes)
    passed = not reason_codes
    return {
        "status": "passed" if passed else "failed",
        "reason_codes": reason_codes,
        "next_required_change": PASS_NEXT_REQUIRED_CHANGE if passed else next_required_change,
    }


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    config: dict[str, Any],
    stage7_summary_path: Path,
    stage7_summary: dict[str, Any] | None,
    network_result: dict[str, Any],
    forward_audit: dict[str, Any],
    metadata_audit: dict[str, Any],
    param_latency_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    metadata = network_result.get("metadata", {})
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_architecture_contrast_summary": str(stage7_summary_path),
        "source_architecture_contrast_status": stage7_summary.get("status") if isinstance(stage7_summary, dict) else None,
        "architecture": ARCHITECTURE,
        "candidate_count": network_result.get("candidate_count", 0),
        "edge_count": network_result.get("edge_count", 0),
        "candidate_graph_encoder_used": bool(metadata.get("candidate_graph_encoder_used", False)),
        "topology_bias_used": bool(metadata.get("topology_bias_used", False)),
        "coverage_memory_token_used": bool(metadata.get("coverage_memory_token_used", False)),
        "roi_budget_fusion_used": bool(metadata.get("roi_budget_fusion_used", False)),
        "masked_logits_valid": forward_audit.get("masked_logits_valid", False),
        "value_head_valid": forward_audit.get("value_head_valid", False),
        "metadata_audit_passed": metadata_audit.get("metadata_audit_passed", False),
        "parameter_count": param_latency_audit.get("parameter_count", 0),
        "median_forward_latency_ms": param_latency_audit.get("median_forward_latency_ms", 0.0),
        "parameter_latency_audit_passed": param_latency_audit.get("parameter_latency_audit_passed", False),
        "boundary_audit_passed": boundary_audit.get("boundary_audit_passed", False),
        "full_network_v1_passed": decision["status"] == "passed",
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


def _boundary_audit(stage7_summary: dict[str, Any] | None) -> dict[str, Any]:
    observed = {}
    violations = []
    if isinstance(stage7_summary, dict):
        for field in BOUNDARY_FIELDS:
            value = bool(stage7_summary.get(field, False))
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
    forward_audit: dict[str, Any],
    metadata_audit: dict[str, Any],
    param_latency_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "masked_logits_valid": forward_audit.get("masked_logits_valid", False),
        "value_head_valid": forward_audit.get("value_head_valid", False),
        "metadata_audit_passed": metadata_audit.get("metadata_audit_passed", False),
        "parameter_latency_audit_passed": param_latency_audit.get("parameter_latency_audit_passed", False),
        "boundary_audit_passed": boundary_audit.get("boundary_audit_passed", False),
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    lines = [
        "# Xunce Full Network v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- architecture: `{summary['architecture']}`",
        f"- parameter_count: `{summary['parameter_count']}`",
        f"- median_forward_latency_ms: `{summary['median_forward_latency_ms']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        "",
        "## Boundary",
        "",
        "- This is a research-only network implementation and forward audit.",
        "- It is not registered as a production/default-policy architecture.",
        "- It does not train PPO, publish checkpoints, replace default policy, or connect an executor.",
        "",
        "## Rejection Report",
        "",
        f"- masked_logits_valid: `{rejection_report['masked_logits_valid']}`",
        f"- value_head_valid: `{rejection_report['value_head_valid']}`",
        f"- metadata_audit_passed: `{rejection_report['metadata_audit_passed']}`",
        f"- parameter_latency_audit_passed: `{rejection_report['parameter_latency_audit_passed']}`",
        f"- boundary_audit_passed: `{rejection_report['boundary_audit_passed']}`",
        "",
    ]
    return "\n".join(lines)


def _empty_network_result() -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "edge_count": 0,
        "metadata": {},
        "forward_audit": {
            "schema_version": FORWARD_AUDIT_SCHEMA_VERSION,
            "masked_logits_valid": False,
            "value_head_valid": False,
        },
        "metadata_audit": {
            "schema_version": METADATA_AUDIT_SCHEMA_VERSION,
            "metadata_audit_passed": False,
        },
        "parameter_latency_audit": {
            "schema_version": PARAM_LATENCY_AUDIT_SCHEMA_VERSION,
            "parameter_latency_audit_passed": False,
        },
        "logit_rows": [],
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        rows.append(payload)
    return rows


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
