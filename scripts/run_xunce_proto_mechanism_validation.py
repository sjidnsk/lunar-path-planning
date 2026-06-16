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
    from xunce_topology_graph_proto_common import TopologyAwareCoverageGraphPrototype
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from scripts.global_99_governance_common import global_99_boundary_defaults
    from scripts.xunce_topology_graph_proto_common import TopologyAwareCoverageGraphPrototype


CONFIG_SCHEMA_VERSION = "xunce-proto-mechanism-validation-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-proto-mechanism-validation-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-proto-mechanism-validation-manifest/v1"
SIGNAL_AUDIT_SCHEMA_VERSION = "xunce-proto-mechanism-validation-signal-audit/v1"
MASK_AUDIT_SCHEMA_VERSION = "xunce-proto-mechanism-validation-mask-audit/v1"
FALLBACK_AUDIT_SCHEMA_VERSION = "xunce-proto-mechanism-validation-fallback-audit/v1"
LOGIT_ROW_SCHEMA_VERSION = "xunce-proto-mechanism-validation-logit-comparison-row/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-proto-mechanism-validation-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-proto-mechanism-validation-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_proto_mechanism_validation_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_proto_mechanism_validation_v1"

SUMMARY_FILE = "xunce-proto-mechanism-validation-summary.json"
MANIFEST_FILE = "xunce-proto-mechanism-validation-manifest.json"
SIGNAL_AUDIT_FILE = "xunce-proto-mechanism-validation-signal-audit.json"
MASK_AUDIT_FILE = "xunce-proto-mechanism-validation-mask-audit.json"
FALLBACK_AUDIT_FILE = "xunce-proto-mechanism-validation-fallback-audit.json"
LOGIT_COMPARISON_FILE = "xunce-proto-mechanism-validation-logit-comparison.jsonl"
BOUNDARY_AUDIT_FILE = "xunce-proto-mechanism-validation-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-proto-mechanism-validation-rejection-report.json"
REPORT_FILE = "xunce-proto-mechanism-validation-report.md"

ARCHITECTURE = "topology_aware_coverage_graph_proto_v1"
PASS_NEXT_REQUIRED_CHANGE = "architecture_contrast_evaluation"
FIX_STAGE5_NEXT_REQUIRED_CHANGE = "fix_xunce_topology_graph_proto"
FAIL_NEXT_REQUIRED_CHANGE = "fix_xunce_proto_mechanism_validation"

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
    parser = argparse.ArgumentParser(description="Run Xunce Prototype Mechanism Validation v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_proto_mechanism_validation(
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
                "edge_rank_changed": summary["edge_rank_changed"],
                "memory_rank_changed": summary["memory_rank_changed"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_proto_mechanism_validation(
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
    stage5_root = resolve_path(Path(config["source_topology_graph_proto_root"]), repo_root)
    stage4_root = resolve_path(Path(config["source_topology_feature_extraction_root"]), repo_root)
    stage5_summary_path = stage5_root / "xunce-topology-graph-proto-summary.json"
    stage5_summary = _load_json(stage5_summary_path)
    boundary_audit = _boundary_audit(stage5_summary)
    source_reason_codes: list[str] = []
    result = _empty_result()

    if _stage5_ready(stage5_summary):
        candidates = _load_jsonl(stage4_root / "xunce-topology-feature-extraction-candidates.jsonl")
        edges = _load_jsonl(stage4_root / "xunce-topology-feature-extraction-edges.jsonl")
        memory = _load_json(stage4_root / "xunce-topology-feature-extraction-memory.json")
        if not candidates:
            source_reason_codes.append("missing_topology_feature_candidates")
        if not edges:
            source_reason_codes.append("missing_topology_feature_edges")
        if not isinstance(memory, dict):
            source_reason_codes.append("missing_topology_feature_memory")
        if not source_reason_codes:
            result = _run_mechanism(config, candidates, edges, memory)

    signal_audit = result["signal_audit"]
    mask_audit = result["mask_audit"]
    fallback_audit = result["fallback_audit"]
    decision = _decision(
        stage5_summary=stage5_summary,
        source_reason_codes=source_reason_codes,
        signal_audit=signal_audit,
        mask_audit=mask_audit,
        fallback_audit=fallback_audit,
        boundary_audit=boundary_audit,
    )
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        stage5_summary_path=stage5_summary_path,
        stage4_root=stage4_root,
        config=config,
        result=result,
        signal_audit=signal_audit,
        mask_audit=mask_audit,
        fallback_audit=fallback_audit,
        boundary_audit=boundary_audit,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)
    rejection_report = _rejection_report(decision, signal_audit, mask_audit, fallback_audit, boundary_audit)
    write_json(paths["signal_audit"], signal_audit)
    write_json(paths["mask_audit"], mask_audit)
    write_json(paths["fallback_audit"], fallback_audit)
    write_jsonl(paths["logit_comparison"], result["comparison_rows"])
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
    for key in ("source_topology_graph_proto_root", "source_topology_feature_extraction_root"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    for key in ("hidden_dim", "message_passing_layers", "mechanism_seed"):
        normalized[key] = _positive_int(payload.get(key), key)
    normalized["dropout"] = _nonnegative_float(payload.get("dropout"), "dropout")
    if normalized["dropout"] >= 1.0:
        raise ConfigError("dropout must be < 1.0")
    normalized["min_logit_delta"] = _nonnegative_float(payload.get("min_logit_delta"), "min_logit_delta")
    for key in ("require_edge_rank_change", "require_memory_rank_change"):
        if not isinstance(payload.get(key), bool):
            raise ConfigError(f"{key} must be a boolean")
    mask = payload.get("action_mask_override")
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
        "signal_audit": output_root / SIGNAL_AUDIT_FILE,
        "mask_audit": output_root / MASK_AUDIT_FILE,
        "fallback_audit": output_root / FALLBACK_AUDIT_FILE,
        "logit_comparison": output_root / LOGIT_COMPARISON_FILE,
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
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


def _stage5_ready(stage5_summary: dict[str, Any] | None) -> bool:
    return (
        isinstance(stage5_summary, dict)
        and stage5_summary.get("status") == "passed"
        and stage5_summary.get("next_required_change") == "xunce_proto_mechanism_validation"
    )


def _empty_result() -> dict[str, Any]:
    failed = {"status": "failed", "reason_codes": ["mechanism_forward_not_run"]}
    return {
        "candidate_count": 0,
        "edge_count": 0,
        "comparison_rows": [],
        "signal_audit": {"schema_version": SIGNAL_AUDIT_SCHEMA_VERSION, **failed},
        "mask_audit": {"schema_version": MASK_AUDIT_SCHEMA_VERSION, **failed},
        "fallback_audit": {"schema_version": FALLBACK_AUDIT_SCHEMA_VERSION, **failed},
    }


def _run_mechanism(config: dict[str, Any], candidates: list[dict[str, Any]], edges: list[dict[str, Any]], memory: dict[str, Any]) -> dict[str, Any]:
    candidates = sorted(candidates, key=lambda row: int(row.get("candidate_index", 0)))
    candidate_feature_names = list(candidates[0].get("features", {}).keys())
    edge_feature_names = list(edges[0].get("features", {}).keys())
    memory_feature_names = list(memory.get("features", {}).keys())
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
    torch.manual_seed(int(config["mechanism_seed"]))
    network = TopologyAwareCoverageGraphPrototype(
        candidate_feature_count=len(candidate_feature_names),
        edge_feature_count=len(edge_feature_names),
        memory_feature_count=len(memory_feature_names),
        hidden_dim=int(config["hidden_dim"]),
        message_passing_layers=int(config["message_passing_layers"]),
        dropout=float(config["dropout"]),
    )
    network.eval()
    with torch.no_grad():
        full = network(
            candidate_features=candidate_features,
            edge_features=edge_features,
            edge_index=edge_index,
            memory_features=memory_features,
            action_mask=action_mask,
        )
        edge_ablated = network(
            candidate_features=candidate_features,
            edge_features=torch.zeros_like(edge_features),
            edge_index=edge_index,
            memory_features=memory_features,
            action_mask=action_mask,
        )
        memory_ablated = network(
            candidate_features=candidate_features,
            edge_features=edge_features,
            edge_index=edge_index,
            memory_features=torch.zeros_like(memory_features),
            action_mask=action_mask,
        )
    full_logits = full.masked_logits[0].detach().cpu()
    edge_logits = edge_ablated.masked_logits[0].detach().cpu()
    memory_logits = memory_ablated.masked_logits[0].detach().cpu()
    full_rank = _ranking(full_logits)
    edge_rank = _ranking(edge_logits)
    memory_rank = _ranking(memory_logits)
    edge_delta = float(torch.max(torch.abs(full_logits - edge_logits)).item())
    memory_delta = float(torch.max(torch.abs(full_logits - memory_logits)).item())
    comparison_rows = [
        {
            "schema_version": LOGIT_ROW_SCHEMA_VERSION,
            "candidate_index": int(candidates[index].get("candidate_index", index)),
            "selected_waypoint": candidates[index].get("selected_waypoint"),
            "action_mask": bool(action_mask[0, index].item()),
            "full_masked_logit": float(full_logits[index].item()),
            "edge_ablated_masked_logit": float(edge_logits[index].item()),
            "memory_ablated_masked_logit": float(memory_logits[index].item()),
            "edge_logit_delta": float(full_logits[index].item() - edge_logits[index].item()),
            "memory_logit_delta": float(full_logits[index].item() - memory_logits[index].item()),
            "full_action_probability": float(full.action_probs[0, index].detach().cpu().item()),
            "fallback_risk": _float(candidates[index].get("features", {}).get("fallback_risk")),
        }
        for index in range(len(candidates))
    ]
    signal_audit = _signal_audit(config, edge_delta, memory_delta, full_rank, edge_rank, memory_rank)
    mask_audit = _mask_audit(action_mask, full.masked_logits[0].detach().cpu(), full.action_probs[0].detach().cpu())
    fallback_audit = _fallback_audit(comparison_rows, full, edge_ablated, memory_ablated)
    return {
        "candidate_count": len(candidates),
        "edge_count": len(edges),
        "comparison_rows": comparison_rows,
        "signal_audit": signal_audit,
        "mask_audit": mask_audit,
        "fallback_audit": fallback_audit,
    }


def _signal_audit(
    config: dict[str, Any],
    edge_delta: float,
    memory_delta: float,
    full_rank: list[int],
    edge_rank: list[int],
    memory_rank: list[int],
) -> dict[str, Any]:
    edge_rank_changed = full_rank != edge_rank
    memory_rank_changed = full_rank != memory_rank
    reasons: list[str] = []
    if edge_delta < float(config["min_logit_delta"]):
        reasons.append("edge_signal_delta_too_small")
    if memory_delta < float(config["min_logit_delta"]):
        reasons.append("memory_signal_delta_too_small")
    if config["require_edge_rank_change"] and not edge_rank_changed:
        reasons.append("edge_signal_rank_unchanged")
    if config["require_memory_rank_change"] and not memory_rank_changed:
        reasons.append("memory_signal_rank_unchanged")
    return {
        "schema_version": SIGNAL_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed",
        "reason_codes": reasons,
        "max_edge_logit_delta": edge_delta,
        "max_memory_logit_delta": memory_delta,
        "full_rank": full_rank,
        "edge_ablated_rank": edge_rank,
        "memory_ablated_rank": memory_rank,
        "edge_rank_changed": edge_rank_changed,
        "memory_rank_changed": memory_rank_changed,
    }


def _mask_audit(action_mask: torch.Tensor, masked_logits: torch.Tensor, action_probs: torch.Tensor) -> dict[str, Any]:
    mask = action_mask[0].detach().cpu()
    invalid_logits = masked_logits[~mask]
    invalid_probs = action_probs[~mask]
    valid_prob_sum = float(action_probs[mask].sum().item())
    reasons: list[str] = []
    if invalid_logits.numel() and bool(torch.any(invalid_logits > -1.0e8).item()):
        reasons.append("invalid_candidate_not_masked")
    if invalid_probs.numel() and bool(torch.any(invalid_probs != 0.0).item()):
        reasons.append("invalid_candidate_probability_nonzero")
    if abs(valid_prob_sum - 1.0) > 1.0e-5:
        reasons.append("valid_action_probability_sum_invalid")
    return {
        "schema_version": MASK_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed",
        "reason_codes": reasons,
        "valid_action_probability_sum": valid_prob_sum,
        "invalid_candidate_count": int((~mask).sum().item()),
    }


def _fallback_audit(rows: list[dict[str, Any]], *outputs: Any) -> dict[str, Any]:
    non_finite_count = 0
    for output in outputs:
        non_finite_count += _non_finite_count(output.logits)
        non_finite_count += _non_finite_count(output.masked_logits, allow_negative_mask=True)
        non_finite_count += _non_finite_count(output.action_probs)
        non_finite_count += _non_finite_count(output.value)
    selected_index = max(rows, key=lambda row: row["full_action_probability"])["candidate_index"] if rows else None
    selected_row = next((row for row in rows if row["candidate_index"] == selected_index), {})
    selected_fallback_risk = _float(selected_row.get("fallback_risk"))
    reasons: list[str] = []
    if non_finite_count:
        reasons.append("non_finite_mechanism_outputs")
    if selected_fallback_risk > 0.0:
        reasons.append("fallback_risk_candidate_selected")
    return {
        "schema_version": FALLBACK_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed",
        "reason_codes": reasons,
        "non_finite_output_count": non_finite_count,
        "selected_candidate_index": selected_index,
        "selected_fallback_risk": selected_fallback_risk,
    }


def _boundary_audit(stage5_summary: dict[str, Any] | None) -> dict[str, Any]:
    violations = []
    if isinstance(stage5_summary, dict):
        for field in BOUNDARY_FIELDS:
            if stage5_summary.get(field) is True:
                violations.append({"source": "topology_graph_proto", "field": field, "value": True})
    return {
        "schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not violations else "failed",
        "reason_codes": [] if not violations else ["proto_mechanism_boundary_violation"],
        "violations": violations,
        **global_99_boundary_defaults(),
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
    }


def _decision(
    *,
    stage5_summary: dict[str, Any] | None,
    source_reason_codes: list[str],
    signal_audit: dict[str, Any],
    mask_audit: dict[str, Any],
    fallback_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    stage5_invalid = False
    if not isinstance(stage5_summary, dict):
        stage5_invalid = True
        reasons.append("missing_topology_graph_proto_summary")
    else:
        if stage5_summary.get("status") != "passed":
            stage5_invalid = True
            reasons.append("topology_graph_proto_not_passed")
        if stage5_summary.get("next_required_change") != "xunce_proto_mechanism_validation":
            stage5_invalid = True
            reasons.append("topology_graph_proto_next_required_change_invalid")
    reasons.extend(source_reason_codes)
    reasons.extend(signal_audit.get("reason_codes", []))
    reasons.extend(mask_audit.get("reason_codes", []))
    reasons.extend(fallback_audit.get("reason_codes", []))
    reasons.extend(boundary_audit.get("reason_codes", []))
    reasons = unique_sorted(reasons)
    status = "passed" if not reasons else "failed"
    if status == "passed":
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif stage5_invalid:
        next_required_change = FIX_STAGE5_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FAIL_NEXT_REQUIRED_CHANGE
    return {"status": status, "reason_codes": reasons, "next_required_change": next_required_change}


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    stage5_summary_path: Path,
    stage4_root: Path,
    config: dict[str, Any],
    result: dict[str, Any],
    signal_audit: dict[str, Any],
    mask_audit: dict[str, Any],
    fallback_audit: dict[str, Any],
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
        "source_topology_graph_proto_summary": str(stage5_summary_path),
        "source_topology_feature_extraction_root": str(stage4_root),
        "architecture": config["architecture"],
        "candidate_count": result["candidate_count"],
        "edge_count": result["edge_count"],
        "mechanism_seed": config["mechanism_seed"],
        "max_edge_logit_delta": signal_audit.get("max_edge_logit_delta", 0.0),
        "max_memory_logit_delta": signal_audit.get("max_memory_logit_delta", 0.0),
        "edge_rank_changed": signal_audit.get("edge_rank_changed", False),
        "memory_rank_changed": signal_audit.get("memory_rank_changed", False),
        "signal_audit_passed": signal_audit["status"] == "passed",
        "mask_audit_passed": mask_audit["status"] == "passed",
        "fallback_audit_passed": fallback_audit["status"] == "passed",
        "boundary_audit_passed": boundary_audit["status"] == "passed",
        "checkpoint_written": False,
        "publishes_checkpoint": False,
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
    signal_audit: dict[str, Any],
    mask_audit: dict[str, Any],
    fallback_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "status": "passed" if decision["status"] == "passed" else "failed",
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "signal_rejections": signal_audit.get("reason_codes", []),
        "mask_rejections": mask_audit.get("reason_codes", []),
        "fallback_rejections": fallback_audit.get("reason_codes", []),
        "boundary_rejections": boundary_audit.get("violations", []),
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Prototype Mechanism Validation v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- max_edge_logit_delta: `{summary['max_edge_logit_delta']}`",
            f"- max_memory_logit_delta: `{summary['max_memory_logit_delta']}`",
            f"- edge_rank_changed: `{summary['edge_rank_changed']}`",
            f"- memory_rank_changed: `{summary['memory_rank_changed']}`",
            "",
            "## Rejections",
            "",
            json.dumps(rejection_report, ensure_ascii=False, indent=2),
            "",
        ]
    )


def _ranking(masked_logits: torch.Tensor) -> list[int]:
    return [int(index) for index in torch.argsort(masked_logits, descending=True).tolist()]


def _action_mask(config: dict[str, Any], candidate_count: int) -> torch.Tensor:
    mask = config["action_mask_override"]
    if len(mask) != candidate_count:
        raise ConfigError("action_mask_override length must match candidate count")
    return torch.tensor([mask], dtype=torch.bool)


def _non_finite_count(tensor: torch.Tensor, *, allow_negative_mask: bool = False) -> int:
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
