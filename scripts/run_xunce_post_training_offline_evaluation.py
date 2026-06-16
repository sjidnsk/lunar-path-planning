from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json
    from global_99_governance_common import global_99_boundary_defaults
    from run_xunce_controlled_training_candidate import _synthetic_batch
    from xunce_full_network_common import XunceFullNetworkV1, parameter_count
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json
    from scripts.global_99_governance_common import global_99_boundary_defaults
    from scripts.run_xunce_controlled_training_candidate import _synthetic_batch
    from scripts.xunce_full_network_common import XunceFullNetworkV1, parameter_count


CONFIG_SCHEMA_VERSION = "xunce-post-training-offline-evaluation-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-post-training-offline-evaluation-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-post-training-offline-evaluation-manifest/v1"
RESULT_ROW_SCHEMA_VERSION = "xunce-post-training-offline-evaluation-result-row/v1"
CHECKPOINT_LOAD_AUDIT_SCHEMA_VERSION = "xunce-post-training-checkpoint-load-audit/v1"
POLICY_DELTA_AUDIT_SCHEMA_VERSION = "xunce-post-training-policy-delta-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-post-training-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-post-training-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_post_training_offline_evaluation_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_post_training_offline_evaluation_v1"

SUMMARY_FILE = "xunce-post-training-offline-evaluation-summary.json"
MANIFEST_FILE = "xunce-post-training-offline-evaluation-manifest.json"
RESULTS_FILE = "xunce-post-training-offline-evaluation-results.jsonl"
CHECKPOINT_LOAD_AUDIT_FILE = "xunce-post-training-checkpoint-load-audit.json"
POLICY_DELTA_AUDIT_FILE = "xunce-post-training-policy-delta-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-post-training-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-post-training-rejection-report.json"
REPORT_FILE = "xunce-post-training-offline-evaluation-report.md"

ARCHITECTURE = "xunce_full_network_v1"
PASS_NEXT_REQUIRED_CHANGE = "shadow_replay_validation"
FAIL_NEXT_REQUIRED_CHANGE = "fix_post_training_offline_evaluation"
FIX_TRAINING_NEXT_REQUIRED_CHANGE = "fix_controlled_training_candidate"
FIX_CHECKPOINT_NEXT_REQUIRED_CHANGE = "fix_controlled_training_candidate_checkpoint"
BOUNDARY_NEXT_REQUIRED_CHANGE = "resolve_xunce_post_training_offline_boundary_rejections"

SOURCE_SUMMARY_FILE = "xunce-controlled-training-candidate-summary.json"
SOURCE_METADATA_FILE = "xunce-controlled-training-checkpoint-metadata.json"
SOURCE_CHECKPOINT_FILE = "xunce-controlled-training-candidate.pt"
SOURCE_PASS_NEXT_REQUIRED_CHANGE = "post_training_offline_evaluation"

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
    parser = argparse.ArgumentParser(description="Run Xunce Post-Training Offline Evaluation v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_post_training_offline_evaluation(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"], "next_required_change": summary["next_required_change"], "summary": summary["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_post_training_offline_evaluation(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    source_root = resolve_path(Path(config["source_controlled_training_root"]), repo_root)
    source_summary = _load_json(source_root / SOURCE_SUMMARY_FILE)
    source_metadata = _load_json(source_root / SOURCE_METADATA_FILE)
    checkpoint_path = source_root / SOURCE_CHECKPOINT_FILE
    source_audit = _source_audit(source_summary)
    boundary_audit = _boundary_audit(source_summary)
    checkpoint_audit: dict[str, Any]
    policy_delta_audit: dict[str, Any]
    result_rows: list[dict[str, Any]] = []
    trained_model: XunceFullNetworkV1 | None = None
    if source_audit["source_controlled_training_audit_passed"] and boundary_audit["source_boundary_audit_passed"]:
        checkpoint_audit, trained_model = _load_checkpoint(checkpoint_path, config, source_metadata)
        if checkpoint_audit["checkpoint_loaded"]:
            result_rows, policy_delta_audit = _evaluate_checkpoint(config, trained_model)
        else:
            policy_delta_audit = _empty_policy_delta_audit()
    else:
        checkpoint_audit = _checkpoint_not_loaded_audit(checkpoint_path, "source_blocked")
        policy_delta_audit = _empty_policy_delta_audit()

    decision = _decision(source_audit, boundary_audit, checkpoint_audit, policy_delta_audit)
    generated_at = utc_now()
    summary = _summary(
        generated_at,
        config_path,
        output_root,
        paths,
        source_summary,
        source_audit,
        boundary_audit,
        checkpoint_audit,
        policy_delta_audit,
        decision,
        repo_root,
    )
    manifest = {"schema_version": MANIFEST_SCHEMA_VERSION, "generated_at": generated_at, "config": str(config_path), "output_root": str(output_root), "artifacts": {key: str(path) for key, path in paths.items()}, "summary_status": summary["status"], "next_required_change": summary["next_required_change"]}
    rejection_report = {"schema_version": REJECTION_REPORT_SCHEMA_VERSION, "status": decision["status"], "reason_codes": decision["reason_codes"], "next_required_change": decision["next_required_change"], "source_controlled_training_audit_passed": source_audit["source_controlled_training_audit_passed"], "checkpoint_loaded": checkpoint_audit["checkpoint_loaded"], "policy_delta_audit_passed": policy_delta_audit["policy_delta_audit_passed"]}
    _write_jsonl(paths["results"], result_rows)
    write_json(paths["checkpoint_load_audit"], checkpoint_audit)
    write_json(paths["policy_delta_audit"], policy_delta_audit)
    write_json(paths["boundary_audit"], boundary_audit)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
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
    if not isinstance(normalized.get("source_controlled_training_root"), str) or not normalized["source_controlled_training_root"].strip():
        raise ConfigError("source_controlled_training_root must be a non-empty string")
    for key in (
        "seed",
        "candidate_feature_count",
        "edge_feature_count",
        "memory_feature_count",
        "context_feature_count",
        "missing_indicator_count",
        "hidden_dim",
        "message_passing_layers",
        "candidate_count",
        "evaluation_case_count",
    ):
        normalized[key] = _int_value(normalized.get(key), key)
    for key in ("candidate_feature_count", "edge_feature_count", "memory_feature_count", "context_feature_count", "hidden_dim", "message_passing_layers", "candidate_count", "evaluation_case_count"):
        if normalized[key] <= 0:
            raise ConfigError(f"{key} must be positive")
    if normalized["missing_indicator_count"] < 0:
        raise ConfigError("missing_indicator_count must be non-negative")
    normalized["min_target_probability_delta"] = _float_value(normalized.get("min_target_probability_delta"), "min_target_probability_delta")
    if normalized["min_target_probability_delta"] < 0.0:
        raise ConfigError("min_target_probability_delta must be non-negative")
    return normalized


def _source_audit(source_summary: dict[str, Any] | None) -> dict[str, Any]:
    reason_codes: list[str] = []
    if not isinstance(source_summary, dict):
        reason_codes.append("missing_controlled_training_summary")
    else:
        if source_summary.get("status") != "passed":
            reason_codes.append("controlled_training_candidate_not_passed")
        if source_summary.get("next_required_change") != SOURCE_PASS_NEXT_REQUIRED_CHANGE:
            reason_codes.append("controlled_training_wrong_next_required_change")
        if source_summary.get("writes_research_checkpoint") is not True:
            reason_codes.append("research_checkpoint_not_declared")
        if source_summary.get("publishes_checkpoint") is True:
            reason_codes.append("controlled_training_published_checkpoint")
    return {
        "schema_version": "xunce-post-training-source-audit/v1",
        "source_status": source_summary.get("status") if isinstance(source_summary, dict) else None,
        "source_next_required_change": source_summary.get("next_required_change") if isinstance(source_summary, dict) else None,
        "reason_codes": unique_sorted(reason_codes),
        "source_controlled_training_audit_passed": not reason_codes,
    }


def _boundary_audit(source_summary: dict[str, Any] | None) -> dict[str, Any]:
    violations: list[str] = []
    observed: dict[str, bool] = {}
    if isinstance(source_summary, dict):
        for field in BOUNDARY_FIELDS:
            value = bool(source_summary.get(field, False))
            observed[field] = value
            if value:
                violations.append(field)
    return {
        "schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION,
        "observed_source_boundary_fields": observed,
        "violating_source_boundary_fields": unique_sorted(violations),
        "source_boundary_audit_passed": not violations,
        "boundary_audit_passed": not violations,
        **_closed_boundary_fields(),
    }


def _load_checkpoint(checkpoint_path: Path, config: dict[str, Any], source_metadata: dict[str, Any] | None) -> tuple[dict[str, Any], XunceFullNetworkV1 | None]:
    reason_codes: list[str] = []
    if not checkpoint_path.is_file():
        reason_codes.append("missing_research_checkpoint")
        return _checkpoint_not_loaded_audit(checkpoint_path, "missing_research_checkpoint", reason_codes), None
    if isinstance(source_metadata, dict) and source_metadata.get("publishes_checkpoint") is True:
        reason_codes.append("source_metadata_published_checkpoint")
    model = _build_model(config)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - exercised by invalid fixture variants
        reason_codes.append("invalid_research_checkpoint")
        audit = _checkpoint_not_loaded_audit(checkpoint_path, f"load_error:{type(exc).__name__}", reason_codes)
        return audit, None
    if not isinstance(payload, dict):
        reason_codes.append("invalid_research_checkpoint")
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        reason_codes.append("checkpoint_state_dict_missing")
    metadata = payload.get("metadata") if isinstance(payload, dict) else {}
    if isinstance(metadata, dict) and metadata.get("architecture") not in (None, ARCHITECTURE):
        reason_codes.append("checkpoint_architecture_mismatch")
    if not reason_codes and isinstance(state_dict, dict):
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:  # pragma: no cover
            reason_codes.append(f"checkpoint_state_dict_load_failed:{type(exc).__name__}")
    passed = not reason_codes
    return {
        "schema_version": CHECKPOINT_LOAD_AUDIT_SCHEMA_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_loaded": passed,
        "checkpoint_read_only": True,
        "checkpoint_schema_valid": passed,
        "reason_codes": unique_sorted(reason_codes),
    }, model if passed else None


def _checkpoint_not_loaded_audit(checkpoint_path: Path, reason: str, reason_codes: list[str] | None = None) -> dict[str, Any]:
    codes = reason_codes or []
    if reason and reason not in codes and not reason.startswith("load_error:") and reason != "source_blocked":
        codes.append(reason)
    return {
        "schema_version": CHECKPOINT_LOAD_AUDIT_SCHEMA_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_loaded": False,
        "checkpoint_read_only": True,
        "checkpoint_schema_valid": False,
        "reason_codes": unique_sorted(codes),
    }


def _evaluate_checkpoint(config: dict[str, Any], trained_model: XunceFullNetworkV1) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    torch.manual_seed(config["seed"])
    fresh_model = _build_model(config)
    target_action = torch.tensor([0], dtype=torch.long)
    target_index = 0
    fresh_model.eval()
    trained_model.eval()
    fresh_latencies: list[float] = []
    trained_latencies: list[float] = []
    for case_index in range(config["evaluation_case_count"]):
        batch = _synthetic_batch(config)
        with torch.no_grad():
            start = time.perf_counter()
            fresh_output = fresh_model(**batch)
            fresh_latencies.append((time.perf_counter() - start) * 1000.0)
            start = time.perf_counter()
            trained_output = trained_model(**batch)
            trained_latencies.append((time.perf_counter() - start) * 1000.0)
            fresh_loss = F.cross_entropy(fresh_output.masked_logits, target_action)
            trained_loss = F.cross_entropy(trained_output.masked_logits, target_action)
            fresh_probability = fresh_output.action_probs[0, target_index]
            trained_probability = trained_output.action_probs[0, target_index]
            fresh_logit = fresh_output.logits[0, target_index]
            trained_logit = trained_output.logits[0, target_index]
        rows.append(
            {
                "schema_version": RESULT_ROW_SCHEMA_VERSION,
                "case_id": f"contract-case-{case_index:03d}",
                "target_action": target_index,
                "fresh_loss": float(fresh_loss.item()),
                "trained_loss": float(trained_loss.item()),
                "loss_delta": float(fresh_loss.item() - trained_loss.item()),
                "fresh_target_probability": float(fresh_probability.item()),
                "trained_target_probability": float(trained_probability.item()),
                "target_probability_delta": float((trained_probability - fresh_probability).item()),
                "fresh_target_logit": float(fresh_logit.item()),
                "trained_target_logit": float(trained_logit.item()),
                "target_logit_delta": float((trained_logit - fresh_logit).item()),
                "fresh_latency_ms": fresh_latencies[-1],
                "trained_latency_ms": trained_latencies[-1],
                "finite_outputs": bool(torch.isfinite(fresh_output.logits).all() and torch.isfinite(trained_output.logits).all() and torch.isfinite(fresh_output.value).all() and torch.isfinite(trained_output.value).all()),
            }
        )
    mean_loss_delta = sum(row["loss_delta"] for row in rows) / len(rows)
    mean_probability_delta = sum(row["target_probability_delta"] for row in rows) / len(rows)
    mean_logit_delta = sum(row["target_logit_delta"] for row in rows) / len(rows)
    finite_outputs = all(row["finite_outputs"] for row in rows)
    trained_loss_lower_than_fresh = all(row["trained_loss"] < row["fresh_loss"] for row in rows)
    target_probability_improved = mean_probability_delta > config["min_target_probability_delta"]
    reason_codes: list[str] = []
    if not finite_outputs:
        reason_codes.append("non_finite_offline_outputs")
    if not trained_loss_lower_than_fresh:
        reason_codes.append("post_training_metric_regression")
    if not target_probability_improved:
        reason_codes.append("post_training_metric_regression")
    return rows, {
        "schema_version": POLICY_DELTA_AUDIT_SCHEMA_VERSION,
        "evaluation_case_count": len(rows),
        "passed_case_count": sum(1 for row in rows if row["finite_outputs"] and row["trained_loss"] < row["fresh_loss"]),
        "failed_case_count": sum(1 for row in rows if not (row["finite_outputs"] and row["trained_loss"] < row["fresh_loss"])),
        "fresh_loss_mean": sum(row["fresh_loss"] for row in rows) / len(rows),
        "trained_loss_mean": sum(row["trained_loss"] for row in rows) / len(rows),
        "mean_loss_delta": mean_loss_delta,
        "mean_target_probability_delta": mean_probability_delta,
        "mean_target_logit_delta": mean_logit_delta,
        "fresh_latency_ms_mean": sum(fresh_latencies) / len(fresh_latencies),
        "trained_latency_ms_mean": sum(trained_latencies) / len(trained_latencies),
        "parameter_count": parameter_count(trained_model),
        "finite_outputs": finite_outputs,
        "trained_loss_lower_than_fresh": trained_loss_lower_than_fresh,
        "target_probability_improved": target_probability_improved,
        "reason_codes": unique_sorted(reason_codes),
        "policy_delta_audit_passed": not reason_codes,
    }


def _empty_policy_delta_audit() -> dict[str, Any]:
    return {
        "schema_version": POLICY_DELTA_AUDIT_SCHEMA_VERSION,
        "evaluation_case_count": 0,
        "passed_case_count": 0,
        "failed_case_count": 0,
        "fresh_loss_mean": None,
        "trained_loss_mean": None,
        "mean_loss_delta": None,
        "mean_target_probability_delta": None,
        "mean_target_logit_delta": None,
        "fresh_latency_ms_mean": None,
        "trained_latency_ms_mean": None,
        "parameter_count": None,
        "finite_outputs": False,
        "trained_loss_lower_than_fresh": False,
        "target_probability_improved": False,
        "reason_codes": [],
        "policy_delta_audit_passed": False,
    }


def _decision(source_audit: dict[str, Any], boundary_audit: dict[str, Any], checkpoint_audit: dict[str, Any], policy_delta_audit: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    reason_codes.extend(source_audit["reason_codes"])
    if not boundary_audit["source_boundary_audit_passed"]:
        reason_codes.append("post_training_source_boundary_violation")
    reason_codes.extend(checkpoint_audit["reason_codes"])
    if checkpoint_audit["checkpoint_loaded"]:
        reason_codes.extend(policy_delta_audit["reason_codes"])
    reason_codes = unique_sorted(reason_codes)
    if not reason_codes:
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif "post_training_source_boundary_violation" in reason_codes:
        next_required_change = BOUNDARY_NEXT_REQUIRED_CHANGE
    elif any(code in reason_codes for code in ("missing_research_checkpoint", "invalid_research_checkpoint", "checkpoint_state_dict_missing", "checkpoint_architecture_mismatch")) or any(code.startswith("checkpoint_state_dict_load_failed") for code in reason_codes):
        next_required_change = FIX_CHECKPOINT_NEXT_REQUIRED_CHANGE
    elif any(code.startswith("missing_controlled_training") or code.startswith("controlled_training") or code == "research_checkpoint_not_declared" for code in reason_codes):
        next_required_change = FIX_TRAINING_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FAIL_NEXT_REQUIRED_CHANGE
    return {"status": "passed" if not reason_codes else "failed", "reason_codes": reason_codes, "next_required_change": next_required_change}


def _summary(
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    source_summary: dict[str, Any] | None,
    source_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
    checkpoint_audit: dict[str, Any],
    policy_delta_audit: dict[str, Any],
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
        "source_controlled_training_status": source_summary.get("status") if isinstance(source_summary, dict) else None,
        "source_controlled_training_next_required_change": source_summary.get("next_required_change") if isinstance(source_summary, dict) else None,
        "source_controlled_training_audit_passed": source_audit["source_controlled_training_audit_passed"],
        "checkpoint_loaded": checkpoint_audit["checkpoint_loaded"],
        "checkpoint_read_only": checkpoint_audit["checkpoint_read_only"],
        "checkpoint_schema_valid": checkpoint_audit["checkpoint_schema_valid"],
        "evaluation_case_count": policy_delta_audit["evaluation_case_count"],
        "passed_case_count": policy_delta_audit["passed_case_count"],
        "failed_case_count": policy_delta_audit["failed_case_count"],
        "fresh_loss_mean": policy_delta_audit["fresh_loss_mean"],
        "trained_loss_mean": policy_delta_audit["trained_loss_mean"],
        "mean_loss_delta": policy_delta_audit["mean_loss_delta"],
        "mean_target_probability_delta": policy_delta_audit["mean_target_probability_delta"],
        "mean_target_logit_delta": policy_delta_audit["mean_target_logit_delta"],
        "parameter_count": policy_delta_audit["parameter_count"],
        "fresh_latency_ms_mean": policy_delta_audit["fresh_latency_ms_mean"],
        "trained_latency_ms_mean": policy_delta_audit["trained_latency_ms_mean"],
        "trained_loss_lower_than_fresh": policy_delta_audit["trained_loss_lower_than_fresh"],
        "target_probability_improved": policy_delta_audit["target_probability_improved"],
        "post_training_offline_evaluation_passed": decision["status"] == "passed",
        "boundary_audit_passed": boundary_audit["boundary_audit_passed"],
        "source_boundary_audit_passed": boundary_audit["source_boundary_audit_passed"],
        "checkpoint_load_audit_passed": checkpoint_audit["checkpoint_loaded"],
        "policy_delta_audit_passed": policy_delta_audit["policy_delta_audit_passed"],
        "uses_research_checkpoint": checkpoint_audit["checkpoint_loaded"],
        "runs_new_training_update": False,
        "runs_new_ppo_update": False,
        "next_required_change": decision["next_required_change"],
        "git_provenance": {"current": git_snapshot(repo_root)},
    }
    summary.update(_closed_boundary_fields())
    return summary


def _build_model(config: dict[str, Any]) -> XunceFullNetworkV1:
    torch.manual_seed(config["seed"])
    return XunceFullNetworkV1(
        candidate_feature_count=config["candidate_feature_count"],
        edge_feature_count=config["edge_feature_count"],
        memory_feature_count=config["memory_feature_count"],
        context_feature_count=config["context_feature_count"],
        missing_indicator_count=config["missing_indicator_count"],
        hidden_dim=config["hidden_dim"],
        message_passing_layers=config["message_passing_layers"],
        dropout=0.0,
    )


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "results": output_root / RESULTS_FILE,
        "checkpoint_load_audit": output_root / CHECKPOINT_LOAD_AUDIT_FILE,
        "policy_delta_audit": output_root / POLICY_DELTA_AUDIT_FILE,
        "boundary_audit": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
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
            "runs_new_training_update": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
        }
    )
    return fields


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Post-Training Offline Evaluation v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- checkpoint_loaded: `{summary['checkpoint_loaded']}`",
            f"- trained_loss_lower_than_fresh: `{summary['trained_loss_lower_than_fresh']}`",
            f"- target_probability_improved: `{summary['target_probability_improved']}`",
            f"- publishes_checkpoint: `{summary['publishes_checkpoint']}`",
            f"- runs_new_training_update: `{summary['runs_new_training_update']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            "",
            "This stage evaluates the Stage 13 research checkpoint read-only. It does not train, publish, install, or connect an executor.",
            "",
        ]
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _int_value(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _float_value(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
