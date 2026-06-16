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


CONFIG_SCHEMA_VERSION = "xunce-full-network-ablation-experiments-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-full-network-ablation-experiments-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-full-network-ablation-experiments-manifest/v1"
RESULT_ROW_SCHEMA_VERSION = "xunce-full-network-ablation-result-row/v1"
MODULE_AUDIT_SCHEMA_VERSION = "xunce-full-network-ablation-module-contribution-audit/v1"
RANKING_AUDIT_SCHEMA_VERSION = "xunce-full-network-ablation-ranking-delta-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-full-network-ablation-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-full-network-ablation-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_full_network_ablation_experiments_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_full_network_ablation_experiments_v1"

SUMMARY_FILE = "xunce-full-network-ablation-experiments-summary.json"
MANIFEST_FILE = "xunce-full-network-ablation-experiments-manifest.json"
RESULTS_FILE = "xunce-full-network-ablation-results.jsonl"
MODULE_AUDIT_FILE = "xunce-full-network-ablation-module-contribution-audit.json"
RANKING_AUDIT_FILE = "xunce-full-network-ablation-ranking-delta-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-full-network-ablation-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-full-network-ablation-rejection-report.json"
REPORT_FILE = "xunce-full-network-ablation-experiments-report.md"

ARCHITECTURE = "xunce_full_network_v1"
PASS_NEXT_REQUIRED_CHANGE = "full_network_stress_evaluation"
FAIL_NEXT_REQUIRED_CHANGE = "fix_full_network_ablation_experiments"
FIX_STAGE9_NEXT_REQUIRED_CHANGE = "fix_full_network_static_contract_validation"

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
    parser = argparse.ArgumentParser(description="Run Xunce Full Network Ablation Experiments v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_full_network_ablation_experiments(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"], "next_required_change": summary["next_required_change"], "summary": summary["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_full_network_ablation_experiments(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    stage9_summary_path = resolve_path(Path(config["source_static_contract_root"]), repo_root) / "xunce-full-network-static-contract-validation-summary.json"
    stage9_summary = _load_json(stage9_summary_path)
    boundary_audit = _boundary_audit(stage9_summary)
    source_reason_codes = _source_reason_codes(stage9_summary)
    results: list[dict[str, Any]] = []
    module_audit = _empty_module_audit()
    ranking_audit = _empty_ranking_audit()
    if not source_reason_codes:
        results = _run_ablations(config)
        module_audit = _module_audit(config, results)
        ranking_audit = _ranking_audit(results)
    decision = _decision(source_reason_codes, module_audit, ranking_audit, boundary_audit)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        stage9_summary_path=stage9_summary_path,
        stage9_summary=stage9_summary,
        results=results,
        module_audit=module_audit,
        ranking_audit=ranking_audit,
        boundary_audit=boundary_audit,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)
    rejection_report = _rejection_report(decision, module_audit, ranking_audit, boundary_audit)
    write_jsonl(paths["results"], results)
    write_json(paths["module_audit"], module_audit)
    write_json(paths["ranking_audit"], ranking_audit)
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
    if not isinstance(payload.get("source_static_contract_root"), str) or not payload["source_static_contract_root"].strip():
        raise ConfigError("source_static_contract_root must be a non-empty string")
    for key in ("candidate_count", "edge_count", "candidate_feature_count", "edge_feature_count", "memory_feature_count", "context_feature_count", "hidden_dim", "message_passing_layers", "seed"):
        normalized[key] = _positive_int(payload.get(key), key)
    normalized["missing_indicator_count"] = _nonnegative_int(payload.get("missing_indicator_count"), "missing_indicator_count")
    normalized["dropout"] = _nonnegative_float(payload.get("dropout"), "dropout")
    normalized["min_logit_delta"] = _nonnegative_float(payload.get("min_logit_delta"), "min_logit_delta")
    mask = payload.get("action_mask")
    if not isinstance(mask, list) or len(mask) != normalized["candidate_count"] or any(not isinstance(value, bool) for value in mask) or not any(mask):
        raise ConfigError("action_mask must be a boolean list matching candidate_count and contain at least one true value")
    normalized["action_mask"] = mask
    return normalized


def _run_ablations(config: dict[str, Any]) -> list[dict[str, Any]]:
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
    cases = {
        "full": base,
        "no_topology_edges": {**base, "edge_features": torch.zeros(0, config["edge_feature_count"]), "edge_index": torch.zeros(0, 2, dtype=torch.long)},
        "zero_memory": {**base, "memory_features": torch.zeros_like(base["memory_features"])},
        "zero_context": {**base, "context_features": torch.zeros_like(base["context_features"])},
        "zero_missing_indicators": {**base, "candidate_missing_indicators": torch.zeros_like(base["candidate_missing_indicators"])},
    }
    rows: list[dict[str, Any]] = []
    full_logits: list[float] | None = None
    full_rank: list[int] | None = None
    with torch.no_grad():
        for case_id, tensors in cases.items():
            output = network(**tensors)
            logits = output.masked_logits[0].detach().cpu().tolist()
            probs = output.action_probs[0].detach().cpu().tolist()
            mask = tensors["action_mask"][0].detach().cpu().tolist()
            valid_indices = [index for index, valid in enumerate(mask) if valid]
            rank = sorted(valid_indices, key=lambda index: (-logits[index], index))
            if case_id == "full":
                full_logits = logits
                full_rank = rank
            assert full_logits is not None and full_rank is not None
            valid_deltas = [abs(float(logits[index]) - float(full_logits[index])) for index in valid_indices]
            row = {
                "schema_version": RESULT_ROW_SCHEMA_VERSION,
                "case_id": case_id,
                "max_valid_logit_delta_from_full": max(valid_deltas) if valid_deltas else 0.0,
                "selected_candidate_index": rank[0] if rank else None,
                "selected_candidate_changed": bool(rank and full_rank and rank[0] != full_rank[0]),
                "rank_order_changed": rank != full_rank,
                "valid_action_probability_sum": sum(float(probs[index]) for index in valid_indices),
                "invalid_action_probability_sum": sum(float(probs[index]) for index, valid in enumerate(mask) if not valid),
                "non_finite_output_count": sum(0 if isfinite(float(value)) else 1 for value in logits + probs + [float(output.value[0].detach().cpu())]),
            }
            rows.append(row)
    return rows


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
        "action_mask": torch.tensor([config["action_mask"]], dtype=torch.bool),
        "candidate_missing_indicators": torch.ones(1, candidate_count, config["missing_indicator_count"]),
    }


def _module_audit(config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = {row["case_id"]: float(row.get("max_valid_logit_delta_from_full", 0.0)) for row in results}
    min_delta = config["min_logit_delta"]
    return {
        "schema_version": MODULE_AUDIT_SCHEMA_VERSION,
        "max_topology_logit_delta": deltas.get("no_topology_edges", 0.0),
        "max_memory_logit_delta": deltas.get("zero_memory", 0.0),
        "max_context_logit_delta": deltas.get("zero_context", 0.0),
        "max_missing_indicator_logit_delta": deltas.get("zero_missing_indicators", 0.0),
        "min_logit_delta": min_delta,
        "topology_ablation_effect_detected": deltas.get("no_topology_edges", 0.0) >= min_delta,
        "memory_ablation_effect_detected": deltas.get("zero_memory", 0.0) >= min_delta,
        "context_ablation_effect_detected": deltas.get("zero_context", 0.0) >= min_delta,
        "missing_indicator_ablation_effect_detected": deltas.get("zero_missing_indicators", 0.0) >= min_delta,
    }


def _ranking_audit(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": RANKING_AUDIT_SCHEMA_VERSION,
        "changed_cases": [row["case_id"] for row in results if row.get("rank_order_changed")],
        "selected_candidate_changed_cases": [row["case_id"] for row in results if row.get("selected_candidate_changed")],
        "mask_preserved": all(abs(float(row.get("valid_action_probability_sum", 0.0)) - 1.0) < 1.0e-5 and float(row.get("invalid_action_probability_sum", 0.0)) == 0.0 for row in results),
        "finite_outputs_preserved": all(int(row.get("non_finite_output_count", 1)) == 0 for row in results),
    }


def _decision(source_reason_codes: list[str], module_audit: dict[str, Any], ranking_audit: dict[str, Any], boundary_audit: dict[str, Any]) -> dict[str, Any]:
    reason_codes = list(source_reason_codes)
    source_blocked = bool(source_reason_codes)
    if not boundary_audit.get("boundary_audit_passed", False):
        reason_codes.append("ablation_boundary_violation")
    if not source_blocked:
        if not all(
            module_audit.get(key, False)
            for key in (
                "topology_ablation_effect_detected",
                "memory_ablation_effect_detected",
                "context_ablation_effect_detected",
                "missing_indicator_ablation_effect_detected",
            )
        ):
            reason_codes.append("ablation_module_contribution_too_small")
        if not ranking_audit.get("mask_preserved", False):
            reason_codes.append("ablation_mask_not_preserved")
        if not ranking_audit.get("finite_outputs_preserved", False):
            reason_codes.append("ablation_non_finite_outputs")
    reason_codes = unique_sorted(reason_codes)
    passed = not reason_codes
    return {
        "status": "passed" if passed else "failed",
        "reason_codes": reason_codes,
        "next_required_change": PASS_NEXT_REQUIRED_CHANGE if passed else (FIX_STAGE9_NEXT_REQUIRED_CHANGE if source_blocked else FAIL_NEXT_REQUIRED_CHANGE),
    }


def _source_reason_codes(stage9_summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(stage9_summary, dict):
        return ["missing_static_contract_validation_summary"]
    reason_codes: list[str] = []
    if stage9_summary.get("status") != "passed":
        reason_codes.append("static_contract_validation_not_passed")
    if stage9_summary.get("next_required_change") != "full_network_ablation_experiments":
        reason_codes.append("static_contract_validation_wrong_next_required_change")
    return reason_codes


def _summary(**kwargs) -> dict[str, Any]:
    generated_at = kwargs["generated_at"]
    config_path = kwargs["config_path"]
    output_root = kwargs["output_root"]
    paths = kwargs["paths"]
    stage9_summary_path = kwargs["stage9_summary_path"]
    stage9_summary = kwargs["stage9_summary"]
    results = kwargs["results"]
    module_audit = kwargs["module_audit"]
    ranking_audit = kwargs["ranking_audit"]
    boundary_audit = kwargs["boundary_audit"]
    decision = kwargs["decision"]
    repo_root = kwargs["repo_root"]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_static_contract_summary": str(stage9_summary_path),
        "source_static_contract_status": stage9_summary.get("status") if isinstance(stage9_summary, dict) else None,
        "architecture": ARCHITECTURE,
        "ablation_case_count": len(results),
        "max_topology_logit_delta": module_audit.get("max_topology_logit_delta", 0.0),
        "max_memory_logit_delta": module_audit.get("max_memory_logit_delta", 0.0),
        "max_context_logit_delta": module_audit.get("max_context_logit_delta", 0.0),
        "max_missing_indicator_logit_delta": module_audit.get("max_missing_indicator_logit_delta", 0.0),
        "topology_ablation_effect_detected": module_audit.get("topology_ablation_effect_detected", False),
        "memory_ablation_effect_detected": module_audit.get("memory_ablation_effect_detected", False),
        "context_ablation_effect_detected": module_audit.get("context_ablation_effect_detected", False),
        "missing_indicator_ablation_effect_detected": module_audit.get("missing_indicator_ablation_effect_detected", False),
        "mask_preserved": ranking_audit.get("mask_preserved", False),
        "finite_outputs_preserved": ranking_audit.get("finite_outputs_preserved", False),
        "boundary_audit_passed": boundary_audit.get("boundary_audit_passed", False),
        "ablation_experiments_passed": decision["status"] == "passed",
        "next_required_change": decision["next_required_change"],
        "git_provenance": {"current": git_snapshot(repo_root)},
    }
    summary.update(_closed_boundary_fields())
    return summary


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "results": output_root / RESULTS_FILE,
        "module_audit": output_root / MODULE_AUDIT_FILE,
        "ranking_audit": output_root / RANKING_AUDIT_FILE,
        "boundary_audit": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _manifest(generated_at: str, config_path: Path, output_root: Path, paths: dict[str, Path], summary: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": MANIFEST_SCHEMA_VERSION, "generated_at": generated_at, "config": str(config_path), "output_root": str(output_root), "artifacts": {key: str(path) for key, path in paths.items()}, "summary_status": summary["status"], "next_required_change": summary["next_required_change"]}


def _boundary_audit(stage9_summary: dict[str, Any] | None) -> dict[str, Any]:
    observed = {}
    violations = []
    if isinstance(stage9_summary, dict):
        for field in BOUNDARY_FIELDS:
            value = bool(stage9_summary.get(field, False))
            observed[field] = value
            if value:
                violations.append(field)
    return {"schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION, "observed_source_boundary_fields": observed, "violating_source_boundary_fields": sorted(set(violations)), "boundary_audit_passed": not violations, **_closed_boundary_fields()}


def _closed_boundary_fields() -> dict[str, bool | float]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields.update({"real_world_release_approved": False, "real_world_performance_claimed": False, "default_policy_replacement_approved": False, "real_executor_connection_approved": False, "starts_online_canary": False, "canary_traffic_fraction": 0.0, "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False, "runs_new_ppo_update": False, "modifies_network": False, "modifies_action_space": False, "modifies_default_astar": False})
    return fields


def _rejection_report(decision: dict[str, Any], module_audit: dict[str, Any], ranking_audit: dict[str, Any], boundary_audit: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": REJECTION_REPORT_SCHEMA_VERSION, "status": decision["status"], "reason_codes": decision["reason_codes"], "next_required_change": decision["next_required_change"], **module_audit, "mask_preserved": ranking_audit.get("mask_preserved", False), "finite_outputs_preserved": ranking_audit.get("finite_outputs_preserved", False), "boundary_audit_passed": boundary_audit.get("boundary_audit_passed", False)}


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    return "\n".join(["# Xunce Full Network Ablation Experiments v1", "", f"- status: `{summary['status']}`", f"- reason_codes: `{summary['reason_codes']}`", f"- ablation_experiments_passed: `{summary['ablation_experiments_passed']}`", f"- next_required_change: `{summary['next_required_change']}`", "", "This is a deterministic ablation gate only. It does not train, publish, install, or connect an executor.", ""])


def _empty_module_audit() -> dict[str, Any]:
    return {"schema_version": MODULE_AUDIT_SCHEMA_VERSION, "topology_ablation_effect_detected": False, "memory_ablation_effect_detected": False, "context_ablation_effect_detected": False, "missing_indicator_ablation_effect_detected": False}


def _empty_ranking_audit() -> dict[str, Any]:
    return {"schema_version": RANKING_AUDIT_SCHEMA_VERSION, "mask_preserved": False, "finite_outputs_preserved": False}


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
