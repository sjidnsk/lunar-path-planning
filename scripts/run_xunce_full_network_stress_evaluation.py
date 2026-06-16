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


CONFIG_SCHEMA_VERSION = "xunce-full-network-stress-evaluation-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-full-network-stress-evaluation-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-full-network-stress-evaluation-manifest/v1"
CASE_ROW_SCHEMA_VERSION = "xunce-full-network-stress-case-row/v1"
LATENCY_AUDIT_SCHEMA_VERSION = "xunce-full-network-stress-latency-audit/v1"
DETERMINISM_AUDIT_SCHEMA_VERSION = "xunce-full-network-stress-determinism-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-full-network-stress-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-full-network-stress-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_full_network_stress_evaluation_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_full_network_stress_evaluation_v1"

SUMMARY_FILE = "xunce-full-network-stress-evaluation-summary.json"
MANIFEST_FILE = "xunce-full-network-stress-evaluation-manifest.json"
CASE_RESULTS_FILE = "xunce-full-network-stress-case-results.jsonl"
LATENCY_AUDIT_FILE = "xunce-full-network-stress-latency-audit.json"
DETERMINISM_AUDIT_FILE = "xunce-full-network-stress-determinism-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-full-network-stress-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-full-network-stress-rejection-report.json"
REPORT_FILE = "xunce-full-network-stress-evaluation-report.md"

ARCHITECTURE = "xunce_full_network_v1"
PASS_NEXT_REQUIRED_CHANGE = "guarded_training_candidate_preflight"
FAIL_NEXT_REQUIRED_CHANGE = "fix_full_network_stress_evaluation"
FIX_STAGE10_NEXT_REQUIRED_CHANGE = "fix_full_network_ablation_experiments"

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
    parser = argparse.ArgumentParser(description="Run Xunce Full Network Stress Evaluation v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_full_network_stress_evaluation(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"], "next_required_change": summary["next_required_change"], "summary": summary["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_full_network_stress_evaluation(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    stage10_summary_path = resolve_path(Path(config["source_ablation_root"]), repo_root) / "xunce-full-network-ablation-experiments-summary.json"
    stage10_summary = _load_json(stage10_summary_path)
    boundary_audit = _boundary_audit(stage10_summary)
    source_reason_codes = _source_reason_codes(stage10_summary)
    case_rows: list[dict[str, Any]] = []
    latency_audit = _empty_latency_audit()
    determinism_audit = _empty_determinism_audit()
    if not source_reason_codes:
        case_rows, latency_audit, determinism_audit = _run_stress(config)
    decision = _decision(source_reason_codes, case_rows, latency_audit, determinism_audit, boundary_audit)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        stage10_summary_path=stage10_summary_path,
        stage10_summary=stage10_summary,
        case_rows=case_rows,
        latency_audit=latency_audit,
        determinism_audit=determinism_audit,
        boundary_audit=boundary_audit,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)
    rejection_report = _rejection_report(decision, latency_audit, determinism_audit, boundary_audit)
    write_jsonl(paths["case_results"], case_rows)
    write_json(paths["latency_audit"], latency_audit)
    write_json(paths["determinism_audit"], determinism_audit)
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
    if not isinstance(payload.get("source_ablation_root"), str) or not payload["source_ablation_root"].strip():
        raise ConfigError("source_ablation_root must be a non-empty string")
    candidate_counts = payload.get("candidate_counts")
    if not isinstance(candidate_counts, list) or not candidate_counts:
        raise ConfigError("candidate_counts must be a non-empty list")
    normalized["candidate_counts"] = [_positive_int(value, "candidate_counts") for value in candidate_counts]
    for key in ("candidate_feature_count", "edge_feature_count", "memory_feature_count", "context_feature_count", "hidden_dim", "message_passing_layers", "seed", "max_parameter_count"):
        normalized[key] = _positive_int(payload.get(key), key)
    normalized["missing_indicator_count"] = _nonnegative_int(payload.get("missing_indicator_count"), "missing_indicator_count")
    normalized["dropout"] = _nonnegative_float(payload.get("dropout"), "dropout")
    normalized["max_forward_latency_ms"] = _nonnegative_float(payload.get("max_forward_latency_ms"), "max_forward_latency_ms")
    normalized["deterministic_tolerance"] = _nonnegative_float(payload.get("deterministic_tolerance"), "deterministic_tolerance")
    return normalized


def _run_stress(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
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
    param_count = parameter_count(network)
    rows: list[dict[str, Any]] = []
    deterministic_deltas: list[float] = []
    with torch.no_grad():
        for candidate_count in config["candidate_counts"]:
            for mode in ("chain", "dense"):
                tensors = _case_tensors(config, candidate_count, mode)
                started = time.perf_counter()
                first = network(**tensors)
                latency_ms = (time.perf_counter() - started) * 1000.0
                second = network(**tensors)
                delta = float(torch.max(torch.abs(first.masked_logits - second.masked_logits)).item())
                deterministic_deltas.append(delta)
                rows.append(_case_row(candidate_count, mode, tensors, first, latency_ms, delta))
    max_latency = max((float(row["forward_latency_ms"]) for row in rows), default=0.0)
    non_finite = sum(int(row["non_finite_output_count"]) for row in rows)
    mask_preserved = all(row["mask_preserved"] for row in rows)
    latency_audit = {
        "schema_version": LATENCY_AUDIT_SCHEMA_VERSION,
        "parameter_count": param_count,
        "max_parameter_count": config["max_parameter_count"],
        "max_forward_latency_ms_observed": max_latency,
        "max_forward_latency_ms": config["max_forward_latency_ms"],
        "parameter_gate_passed": param_count <= config["max_parameter_count"],
        "latency_gate_passed": max_latency <= config["max_forward_latency_ms"],
    }
    determinism_audit = {
        "schema_version": DETERMINISM_AUDIT_SCHEMA_VERSION,
        "deterministic_replay_max_delta": max(deterministic_deltas, default=0.0),
        "deterministic_tolerance": config["deterministic_tolerance"],
        "deterministic_replay_passed": max(deterministic_deltas, default=0.0) <= config["deterministic_tolerance"],
        "non_finite_output_count": non_finite,
        "mask_preserved": mask_preserved,
        "fallback_rate": 0.0 if rows and non_finite == 0 and mask_preserved else 1.0,
    }
    return rows, latency_audit, determinism_audit


def _case_tensors(config: dict[str, Any], candidate_count: int, mode: str) -> dict[str, Any]:
    if mode == "dense":
        pairs = [(left, right) for left in range(candidate_count) for right in range(left + 1, candidate_count)]
    else:
        pairs = [(index, index + 1) for index in range(candidate_count - 1)]
    edge_index = torch.tensor(pairs, dtype=torch.long) if pairs else torch.zeros(0, 2, dtype=torch.long)
    edge_count = edge_index.shape[0]
    mask = torch.ones(1, candidate_count, dtype=torch.bool)
    if candidate_count > 1:
        mask[0, 1] = False
    missing = torch.zeros(1, candidate_count, config["missing_indicator_count"])
    if candidate_count > 2 and config["missing_indicator_count"] > 0:
        missing[:, 2:, :] = 1.0
    scale = 10.0 if mode == "dense" else 1.0
    return {
        "candidate_features": torch.randn(1, candidate_count, config["candidate_feature_count"]) * scale,
        "edge_features": torch.randn(edge_count, config["edge_feature_count"]) * scale,
        "edge_index": edge_index,
        "memory_features": torch.randn(1, config["memory_feature_count"]) * scale,
        "context_features": torch.randn(1, candidate_count, config["context_feature_count"]) * scale,
        "action_mask": mask,
        "candidate_missing_indicators": missing,
    }


def _case_row(candidate_count: int, mode: str, tensors: dict[str, Any], output, latency_ms: float, replay_delta: float) -> dict[str, Any]:
    logits = output.masked_logits[0].detach().cpu().tolist()
    probs = output.action_probs[0].detach().cpu().tolist()
    mask = tensors["action_mask"][0].detach().cpu().tolist()
    non_finite = sum(0 if isfinite(float(value)) else 1 for value in logits + probs + [float(output.value[0].detach().cpu())])
    return {
        "schema_version": CASE_ROW_SCHEMA_VERSION,
        "case_id": f"{mode}_{candidate_count}",
        "candidate_count": candidate_count,
        "edge_count": int(tensors["edge_index"].shape[0]),
        "mode": mode,
        "forward_latency_ms": latency_ms,
        "deterministic_replay_delta": replay_delta,
        "valid_action_probability_sum": sum(float(prob) for prob, valid in zip(probs, mask) if valid),
        "invalid_action_probability_sum": sum(float(prob) for prob, valid in zip(probs, mask) if not valid),
        "mask_preserved": abs(sum(float(prob) for prob, valid in zip(probs, mask) if valid) - 1.0) < 1.0e-5 and sum(float(prob) for prob, valid in zip(probs, mask) if not valid) == 0.0,
        "non_finite_output_count": non_finite,
    }


def _decision(source_reason_codes: list[str], case_rows: list[dict[str, Any]], latency_audit: dict[str, Any], determinism_audit: dict[str, Any], boundary_audit: dict[str, Any]) -> dict[str, Any]:
    reason_codes = list(source_reason_codes)
    source_blocked = bool(source_reason_codes)
    if not boundary_audit.get("boundary_audit_passed", False):
        reason_codes.append("stress_boundary_violation")
    if not source_blocked:
        if not latency_audit.get("parameter_gate_passed", False):
            reason_codes.append("stress_parameter_gate_failed")
        if not latency_audit.get("latency_gate_passed", False):
            reason_codes.append("stress_latency_gate_failed")
        if not determinism_audit.get("deterministic_replay_passed", False):
            reason_codes.append("stress_deterministic_replay_failed")
        if determinism_audit.get("non_finite_output_count", 1) != 0:
            reason_codes.append("stress_non_finite_outputs")
        if not determinism_audit.get("mask_preserved", False):
            reason_codes.append("stress_mask_not_preserved")
    reason_codes = unique_sorted(reason_codes)
    passed = not reason_codes
    return {"status": "passed" if passed else "failed", "reason_codes": reason_codes, "next_required_change": PASS_NEXT_REQUIRED_CHANGE if passed else (FIX_STAGE10_NEXT_REQUIRED_CHANGE if source_blocked else FAIL_NEXT_REQUIRED_CHANGE)}


def _source_reason_codes(stage10_summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(stage10_summary, dict):
        return ["missing_ablation_experiments_summary"]
    reason_codes: list[str] = []
    if stage10_summary.get("status") != "passed":
        reason_codes.append("ablation_experiments_not_passed")
    if stage10_summary.get("next_required_change") != "full_network_stress_evaluation":
        reason_codes.append("ablation_experiments_wrong_next_required_change")
    return reason_codes


def _summary(**kwargs) -> dict[str, Any]:
    case_rows = kwargs["case_rows"]
    latency_audit = kwargs["latency_audit"]
    determinism_audit = kwargs["determinism_audit"]
    boundary_audit = kwargs["boundary_audit"]
    decision = kwargs["decision"]
    stage10_summary = kwargs["stage10_summary"]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": kwargs["generated_at"],
        "config": str(kwargs["config_path"]),
        "output_root": str(kwargs["output_root"]),
        "summary": str(kwargs["paths"]["summary"]),
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_ablation_summary": str(kwargs["stage10_summary_path"]),
        "source_ablation_status": stage10_summary.get("status") if isinstance(stage10_summary, dict) else None,
        "architecture": ARCHITECTURE,
        "stress_case_count": len(case_rows),
        "max_candidate_count": max((int(row.get("candidate_count", 0)) for row in case_rows), default=0),
        "max_edge_count": max((int(row.get("edge_count", 0)) for row in case_rows), default=0),
        "parameter_count": latency_audit.get("parameter_count", 0),
        "max_forward_latency_ms_observed": latency_audit.get("max_forward_latency_ms_observed", 0.0),
        "non_finite_output_count": determinism_audit.get("non_finite_output_count", 0),
        "mask_preserved": determinism_audit.get("mask_preserved", False),
        "deterministic_replay_max_delta": determinism_audit.get("deterministic_replay_max_delta", 0.0),
        "deterministic_replay_passed": determinism_audit.get("deterministic_replay_passed", False),
        "latency_gate_passed": latency_audit.get("latency_gate_passed", False),
        "parameter_gate_passed": latency_audit.get("parameter_gate_passed", False),
        "fallback_rate": determinism_audit.get("fallback_rate", 1.0),
        "boundary_audit_passed": boundary_audit.get("boundary_audit_passed", False),
        "stress_evaluation_passed": decision["status"] == "passed",
        "next_required_change": decision["next_required_change"],
        "git_provenance": {"current": git_snapshot(kwargs["repo_root"])},
    }
    summary.update(_closed_boundary_fields())
    return summary


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {"summary": output_root / SUMMARY_FILE, "manifest": output_root / MANIFEST_FILE, "case_results": output_root / CASE_RESULTS_FILE, "latency_audit": output_root / LATENCY_AUDIT_FILE, "determinism_audit": output_root / DETERMINISM_AUDIT_FILE, "boundary_audit": output_root / BOUNDARY_AUDIT_FILE, "rejection_report": output_root / REJECTION_REPORT_FILE, "report": output_root / REPORT_FILE}


def _manifest(generated_at: str, config_path: Path, output_root: Path, paths: dict[str, Path], summary: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": MANIFEST_SCHEMA_VERSION, "generated_at": generated_at, "config": str(config_path), "output_root": str(output_root), "artifacts": {key: str(path) for key, path in paths.items()}, "summary_status": summary["status"], "next_required_change": summary["next_required_change"]}


def _boundary_audit(stage10_summary: dict[str, Any] | None) -> dict[str, Any]:
    observed = {}
    violations = []
    if isinstance(stage10_summary, dict):
        for field in BOUNDARY_FIELDS:
            value = bool(stage10_summary.get(field, False))
            observed[field] = value
            if value:
                violations.append(field)
    return {"schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION, "observed_source_boundary_fields": observed, "violating_source_boundary_fields": sorted(set(violations)), "boundary_audit_passed": not violations, **_closed_boundary_fields()}


def _closed_boundary_fields() -> dict[str, bool | float]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields.update({"real_world_release_approved": False, "real_world_performance_claimed": False, "default_policy_replacement_approved": False, "real_executor_connection_approved": False, "starts_online_canary": False, "canary_traffic_fraction": 0.0, "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False, "runs_new_ppo_update": False, "modifies_network": False, "modifies_action_space": False, "modifies_default_astar": False})
    return fields


def _rejection_report(decision: dict[str, Any], latency_audit: dict[str, Any], determinism_audit: dict[str, Any], boundary_audit: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": REJECTION_REPORT_SCHEMA_VERSION, "status": decision["status"], "reason_codes": decision["reason_codes"], "next_required_change": decision["next_required_change"], **latency_audit, **determinism_audit, "boundary_audit_passed": boundary_audit.get("boundary_audit_passed", False)}


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    return "\n".join(["# Xunce Full Network Stress Evaluation v1", "", f"- status: `{summary['status']}`", f"- reason_codes: `{summary['reason_codes']}`", f"- stress_evaluation_passed: `{summary['stress_evaluation_passed']}`", f"- next_required_change: `{summary['next_required_change']}`", "", "This is a stability gate only. It does not train, publish, install, or connect an executor.", ""])


def _empty_latency_audit() -> dict[str, Any]:
    return {"schema_version": LATENCY_AUDIT_SCHEMA_VERSION, "parameter_gate_passed": False, "latency_gate_passed": False}


def _empty_determinism_audit() -> dict[str, Any]:
    return {"schema_version": DETERMINISM_AUDIT_SCHEMA_VERSION, "deterministic_replay_passed": False, "non_finite_output_count": 0, "mask_preserved": False, "fallback_rate": 1.0}


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
