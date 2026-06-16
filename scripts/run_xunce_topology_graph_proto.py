from __future__ import annotations

import argparse
import json
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
    from xunce_topology_graph_proto_common import TopologyAwareCoverageGraphPrototype, parameter_count
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from scripts.global_99_governance_common import global_99_boundary_defaults
    from scripts.xunce_topology_graph_proto_common import TopologyAwareCoverageGraphPrototype, parameter_count


CONFIG_SCHEMA_VERSION = "xunce-topology-graph-proto-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-topology-graph-proto-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-topology-graph-proto-manifest/v1"
FORWARD_AUDIT_SCHEMA_VERSION = "xunce-topology-graph-proto-forward-pass-audit/v1"
SHAPE_AUDIT_SCHEMA_VERSION = "xunce-topology-graph-proto-shape-audit/v1"
PARAMETER_AUDIT_SCHEMA_VERSION = "xunce-topology-graph-proto-parameter-audit/v1"
LOGIT_ROW_SCHEMA_VERSION = "xunce-topology-graph-proto-logit-row/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-topology-graph-proto-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-topology-graph-proto-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_topology_graph_proto_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_topology_graph_proto_v1"

SUMMARY_FILE = "xunce-topology-graph-proto-summary.json"
MANIFEST_FILE = "xunce-topology-graph-proto-manifest.json"
FORWARD_AUDIT_FILE = "xunce-topology-graph-proto-forward-pass-audit.json"
SHAPE_AUDIT_FILE = "xunce-topology-graph-proto-shape-audit.json"
PARAMETER_AUDIT_FILE = "xunce-topology-graph-proto-parameter-audit.json"
LOGITS_FILE = "xunce-topology-graph-proto-logits.jsonl"
BOUNDARY_AUDIT_FILE = "xunce-topology-graph-proto-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-topology-graph-proto-rejection-report.json"
REPORT_FILE = "xunce-topology-graph-proto-report.md"

ARCHITECTURE = "topology_aware_coverage_graph_proto_v1"
PASS_NEXT_REQUIRED_CHANGE = "xunce_proto_mechanism_validation"
FIX_STAGE4_NEXT_REQUIRED_CHANGE = "fix_xunce_topology_feature_extraction_audit"
FAIL_NEXT_REQUIRED_CHANGE = "fix_xunce_topology_graph_proto"

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
    parser = argparse.ArgumentParser(description="Run Xunce Topology Graph Prototype v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_topology_graph_proto(
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
                "candidate_count": summary["candidate_count"],
                "edge_count": summary["edge_count"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_topology_graph_proto(
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
    source_root = resolve_path(Path(config["source_topology_feature_extraction_root"]), repo_root)
    source_paths = _source_paths(source_root)
    stage4_summary = _load_json(source_paths["summary"])
    boundary_audit = _boundary_audit(stage4_summary)

    proto_result = _empty_proto_result()
    source_reason_codes: list[str] = []
    if _stage4_ready(stage4_summary):
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
            proto_result = _run_prototype(config, candidates, edges, memory)

    forward_audit = proto_result["forward_audit"]
    shape_audit = proto_result["shape_audit"]
    parameter_audit = proto_result["parameter_audit"]
    decision = _decision(
        stage4_summary=stage4_summary,
        source_reason_codes=source_reason_codes,
        forward_audit=forward_audit,
        shape_audit=shape_audit,
        parameter_audit=parameter_audit,
        boundary_audit=boundary_audit,
    )
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        source_paths=source_paths,
        config=config,
        proto_result=proto_result,
        forward_audit=forward_audit,
        shape_audit=shape_audit,
        parameter_audit=parameter_audit,
        boundary_audit=boundary_audit,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)
    rejection_report = _rejection_report(decision, forward_audit, shape_audit, parameter_audit, boundary_audit)

    write_json(paths["forward_audit"], forward_audit)
    write_json(paths["shape_audit"], shape_audit)
    write_json(paths["parameter_audit"], parameter_audit)
    write_jsonl(paths["logits"], proto_result["logit_rows"])
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
    for key in ("source_topology_feature_extraction_root",):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    for key in ("hidden_dim", "message_passing_layers", "max_forward_passes", "max_parameter_count", "seed"):
        normalized[key] = _positive_int(payload.get(key), key)
    for key in ("dropout", "max_forward_latency_ms"):
        normalized[key] = _nonnegative_float(payload.get(key), key)
    if normalized["dropout"] >= 1.0:
        raise ConfigError("dropout must be < 1.0")
    if "action_mask_override" in payload:
        mask = payload["action_mask_override"]
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
        "shape_audit": output_root / SHAPE_AUDIT_FILE,
        "parameter_audit": output_root / PARAMETER_AUDIT_FILE,
        "logits": output_root / LOGITS_FILE,
        "boundary_audit": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _source_paths(source_root: Path) -> dict[str, Path]:
    return {
        "summary": source_root / "xunce-topology-feature-extraction-audit-summary.json",
        "candidates": source_root / "xunce-topology-feature-extraction-candidates.jsonl",
        "edges": source_root / "xunce-topology-feature-extraction-edges.jsonl",
        "memory": source_root / "xunce-topology-feature-extraction-memory.json",
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
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(row, dict):
            return []
        rows.append(row)
    return rows


def _stage4_ready(stage4_summary: dict[str, Any] | None) -> bool:
    return (
        isinstance(stage4_summary, dict)
        and stage4_summary.get("status") == "passed"
        and stage4_summary.get("next_required_change") == "topology_aware_coverage_graph_proto"
    )


def _empty_proto_result() -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "edge_count": 0,
        "candidate_feature_count": 0,
        "edge_feature_count": 0,
        "memory_feature_count": 0,
        "candidate_graph_used": False,
        "memory_token_used": False,
        "non_finite_output_count": 0,
        "logit_rows": [],
        "forward_audit": _audit("forward", FORWARD_AUDIT_SCHEMA_VERSION, ["prototype_forward_not_run"]),
        "shape_audit": _audit("shape", SHAPE_AUDIT_SCHEMA_VERSION, ["prototype_forward_not_run"]),
        "parameter_audit": _audit("parameter", PARAMETER_AUDIT_SCHEMA_VERSION, ["prototype_forward_not_run"]),
    }


def _run_prototype(
    config: dict[str, Any],
    candidates: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    memory: dict[str, Any],
) -> dict[str, Any]:
    candidates = sorted(candidates, key=lambda row: int(row.get("candidate_index", 0)))
    candidate_feature_names = list(candidates[0].get("features", {}).keys()) if candidates else []
    edge_feature_names = list(edges[0].get("features", {}).keys()) if edges else []
    memory_feature_names = list(memory.get("features", {}).keys()) if isinstance(memory.get("features"), dict) else []
    candidate_features = torch.tensor(
        [[[_float(row.get("features", {}).get(name)) for name in candidate_feature_names] for row in candidates]],
        dtype=torch.float32,
    )
    edge_features = torch.tensor(
        [[_float(row.get("features", {}).get(name)) for name in edge_feature_names] for row in edges],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [[int(row.get("candidate_left_index", 0)), int(row.get("candidate_right_index", 0))] for row in edges],
        dtype=torch.long,
    )
    memory_features = torch.tensor(
        [[_float(memory.get("features", {}).get(name)) for name in memory_feature_names]],
        dtype=torch.float32,
    )
    action_mask = _action_mask(config, len(candidates))
    torch.manual_seed(int(config["seed"]))
    network = TopologyAwareCoverageGraphPrototype(
        candidate_feature_count=len(candidate_feature_names),
        edge_feature_count=len(edge_feature_names),
        memory_feature_count=len(memory_feature_names),
        hidden_dim=int(config["hidden_dim"]),
        message_passing_layers=int(config["message_passing_layers"]),
        dropout=float(config["dropout"]),
    )
    network.eval()
    durations: list[float] = []
    output = None
    with torch.no_grad():
        for _ in range(int(config["max_forward_passes"])):
            start = time.perf_counter()
            output = network(
                candidate_features=candidate_features,
                edge_features=edge_features,
                edge_index=edge_index,
                memory_features=memory_features,
                action_mask=action_mask,
            )
            durations.append((time.perf_counter() - start) * 1000.0)
    assert output is not None
    logits = output.logits[0].detach().cpu()
    masked_logits = output.masked_logits[0].detach().cpu()
    action_probs = output.action_probs[0].detach().cpu()
    value = output.value.detach().cpu()
    non_finite_count = _non_finite_tensor_count(output.logits) + _non_finite_tensor_count(output.masked_logits, allow_negative_mask=True)
    non_finite_count += _non_finite_tensor_count(output.action_probs) + _non_finite_tensor_count(output.value)
    valid_probability_sum = float(action_probs[action_mask[0].cpu()].sum().item())
    logit_rows = [
        {
            "schema_version": LOGIT_ROW_SCHEMA_VERSION,
            "candidate_index": int(candidates[index].get("candidate_index", index)),
            "selected_waypoint": candidates[index].get("selected_waypoint"),
            "action_mask": bool(action_mask[0, index].item()),
            "logit": float(logits[index].item()),
            "masked_logit": float(masked_logits[index].item()),
            "action_probability": float(action_probs[index].item()),
        }
        for index in range(len(candidates))
    ]
    parameter_total = parameter_count(network)
    average_latency_ms = sum(durations) / len(durations) if durations else 0.0
    forward_audit = _forward_audit(
        output=output,
        candidate_count=len(candidates),
        non_finite_count=non_finite_count,
        valid_probability_sum=valid_probability_sum,
        value=[float(item) for item in value.tolist()],
        candidate_graph_used=bool(edges),
        memory_token_used=bool(memory_feature_names),
    )
    shape_audit = _shape_audit(output=output, candidate_count=len(candidates))
    parameter_audit = _parameter_audit(
        parameter_count_value=parameter_total,
        average_latency_ms=average_latency_ms,
        max_parameter_count=int(config["max_parameter_count"]),
        max_forward_latency_ms=float(config["max_forward_latency_ms"]),
    )
    return {
        "candidate_count": len(candidates),
        "edge_count": len(edges),
        "candidate_feature_count": len(candidate_feature_names),
        "edge_feature_count": len(edge_feature_names),
        "memory_feature_count": len(memory_feature_names),
        "candidate_graph_used": bool(edges),
        "memory_token_used": bool(memory_feature_names),
        "non_finite_output_count": non_finite_count,
        "logit_rows": logit_rows,
        "forward_audit": forward_audit,
        "shape_audit": shape_audit,
        "parameter_audit": parameter_audit,
    }


def _forward_audit(
    *,
    output: Any,
    candidate_count: int,
    non_finite_count: int,
    valid_probability_sum: float,
    value: list[float],
    candidate_graph_used: bool,
    memory_token_used: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    if non_finite_count:
        reasons.append("non_finite_proto_outputs")
    if abs(valid_probability_sum - 1.0) > 1.0e-5:
        reasons.append("invalid_action_probability_sum")
    if not candidate_graph_used:
        reasons.append("candidate_graph_not_used")
    if not memory_token_used:
        reasons.append("memory_token_not_used")
    return {
        "schema_version": FORWARD_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed",
        "reason_codes": reasons,
        "logits_shape": list(output.logits.shape),
        "masked_logits_shape": list(output.masked_logits.shape),
        "action_probs_shape": list(output.action_probs.shape),
        "value_shape": list(output.value.shape),
        "candidate_count": candidate_count,
        "non_finite_output_count": non_finite_count,
        "valid_action_probability_sum": valid_probability_sum,
        "value": value,
        "candidate_graph_used": candidate_graph_used,
        "memory_token_used": memory_token_used,
    }


def _shape_audit(*, output: Any, candidate_count: int) -> dict[str, Any]:
    reasons: list[str] = []
    expected_candidate_shape = [1, candidate_count]
    if list(output.logits.shape) != expected_candidate_shape:
        reasons.append("logits_shape_invalid")
    if list(output.masked_logits.shape) != expected_candidate_shape:
        reasons.append("masked_logits_shape_invalid")
    if list(output.action_probs.shape) != expected_candidate_shape:
        reasons.append("action_probs_shape_invalid")
    if list(output.value.shape) != [1]:
        reasons.append("value_shape_invalid")
    return {
        "schema_version": SHAPE_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed",
        "reason_codes": reasons,
        "expected_candidate_shape": expected_candidate_shape,
        "value_shape": list(output.value.shape),
    }


def _parameter_audit(
    *,
    parameter_count_value: int,
    average_latency_ms: float,
    max_parameter_count: int,
    max_forward_latency_ms: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    if parameter_count_value > max_parameter_count:
        reasons.append("prototype_parameter_count_exceeded")
    if average_latency_ms > max_forward_latency_ms:
        reasons.append("prototype_forward_latency_exceeded")
    return {
        "schema_version": PARAMETER_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed",
        "reason_codes": reasons,
        "parameter_count": parameter_count_value,
        "max_parameter_count": max_parameter_count,
        "average_forward_latency_ms": average_latency_ms,
        "max_forward_latency_ms": max_forward_latency_ms,
        "checkpoint_written": False,
    }


def _boundary_audit(stage4_summary: dict[str, Any] | None) -> dict[str, Any]:
    violations = []
    if isinstance(stage4_summary, dict):
        for field in BOUNDARY_FIELDS:
            if stage4_summary.get(field) is True:
                violations.append({"source": "topology_feature_extraction_audit", "field": field, "value": True})
    return {
        "schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not violations else "failed",
        "reason_codes": [] if not violations else ["topology_graph_proto_boundary_violation"],
        "violations": violations,
        **global_99_boundary_defaults(),
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
    }


def _decision(
    *,
    stage4_summary: dict[str, Any] | None,
    source_reason_codes: list[str],
    forward_audit: dict[str, Any],
    shape_audit: dict[str, Any],
    parameter_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    stage4_invalid = False
    if not isinstance(stage4_summary, dict):
        stage4_invalid = True
        reasons.append("missing_topology_feature_extraction_summary")
    else:
        if stage4_summary.get("status") != "passed":
            stage4_invalid = True
            reasons.append("topology_feature_extraction_not_passed")
        if stage4_summary.get("next_required_change") != "topology_aware_coverage_graph_proto":
            stage4_invalid = True
            reasons.append("topology_feature_extraction_next_required_change_invalid")
    reasons.extend(source_reason_codes)
    reasons.extend(forward_audit.get("reason_codes", []))
    reasons.extend(shape_audit.get("reason_codes", []))
    reasons.extend(parameter_audit.get("reason_codes", []))
    reasons.extend(boundary_audit.get("reason_codes", []))
    reasons = unique_sorted(reasons)
    status = "passed" if not reasons else "failed"
    if status == "passed":
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif stage4_invalid:
        next_required_change = FIX_STAGE4_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FAIL_NEXT_REQUIRED_CHANGE
    return {"status": status, "reason_codes": reasons, "next_required_change": next_required_change}


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    source_paths: dict[str, Path],
    config: dict[str, Any],
    proto_result: dict[str, Any],
    forward_audit: dict[str, Any],
    shape_audit: dict[str, Any],
    parameter_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "source_topology_feature_extraction_summary": str(source_paths["summary"]),
        "architecture": config["architecture"],
        "hidden_dim": config["hidden_dim"],
        "message_passing_layers": config["message_passing_layers"],
        "candidate_count": proto_result["candidate_count"],
        "edge_count": proto_result["edge_count"],
        "candidate_feature_count": proto_result["candidate_feature_count"],
        "edge_feature_count": proto_result["edge_feature_count"],
        "memory_feature_count": proto_result["memory_feature_count"],
        "candidate_graph_used": proto_result["candidate_graph_used"],
        "memory_token_used": proto_result["memory_token_used"],
        "non_finite_output_count": proto_result["non_finite_output_count"],
        "parameter_count": parameter_audit.get("parameter_count", 0),
        "average_forward_latency_ms": parameter_audit.get("average_forward_latency_ms", 0.0),
        "forward_pass_audit_passed": forward_audit["status"] == "passed",
        "shape_audit_passed": shape_audit["status"] == "passed",
        "parameter_audit_passed": parameter_audit["status"] == "passed",
        "boundary_audit_passed": boundary_audit["status"] == "passed",
        "implements_research_prototype": True,
        "publishes_checkpoint": False,
        "checkpoint_written": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "runs_new_ppo_update": False,
        "modifies_network": False,
        "modifies_action_space": False,
        "modifies_default_astar": False,
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "uses_path_planner": False,
        "uses_ppo_policy": False,
        "next_required_change": decision["next_required_change"],
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
    decision: dict[str, Any],
    forward_audit: dict[str, Any],
    shape_audit: dict[str, Any],
    parameter_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "status": "passed" if decision["status"] == "passed" else "failed",
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "forward_rejections": forward_audit.get("reason_codes", []),
        "shape_rejections": shape_audit.get("reason_codes", []),
        "parameter_rejections": parameter_audit.get("reason_codes", []),
        "boundary_rejections": boundary_audit.get("violations", []),
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Topology Graph Prototype v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- architecture: `{summary['architecture']}`",
            f"- candidate_count: `{summary['candidate_count']}`",
            f"- edge_count: `{summary['edge_count']}`",
            f"- parameter_count: `{summary['parameter_count']}`",
            f"- average_forward_latency_ms: `{summary['average_forward_latency_ms']}`",
            "",
            "## Rejections",
            "",
            json.dumps(rejection_report, ensure_ascii=False, indent=2),
            "",
        ]
    )


def _audit(name: str, schema_version: str, reasons: list[str]) -> dict[str, Any]:
    return {"schema_version": schema_version, "status": "passed" if not reasons else "failed", "reason_codes": reasons, "audit": name}


def _action_mask(config: dict[str, Any], candidate_count: int) -> torch.Tensor:
    override = config.get("action_mask_override")
    if override is not None:
        if len(override) != candidate_count:
            raise ConfigError("action_mask_override length must match candidate count")
        return torch.tensor([override], dtype=torch.bool)
    return torch.ones((1, candidate_count), dtype=torch.bool)


def _non_finite_tensor_count(tensor: torch.Tensor, *, allow_negative_mask: bool = False) -> int:
    values = tensor.detach().cpu()
    if allow_negative_mask:
        values = values[values > -1.0e8]
    return int((~torch.isfinite(values)).sum().item())


def _float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if isfinite(parsed) else 0.0


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise ConfigError(f"{label} must be a positive integer")
    return parsed


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be nonnegative")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be nonnegative") from exc
    if not isfinite(parsed) or parsed < 0.0:
        raise ConfigError(f"{label} must be nonnegative")
    return parsed


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
