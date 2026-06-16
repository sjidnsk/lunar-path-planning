from __future__ import annotations

import argparse
import json
import math
import sys
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
    from xunce_full_network_common import XunceFullNetworkV1, parameter_count
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json
    from scripts.global_99_governance_common import global_99_boundary_defaults
    from scripts.xunce_full_network_common import XunceFullNetworkV1, parameter_count


CONFIG_SCHEMA_VERSION = "xunce-controlled-training-candidate-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-controlled-training-candidate-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-controlled-training-candidate-manifest/v1"
LOSS_AUDIT_SCHEMA_VERSION = "xunce-controlled-training-loss-audit/v1"
GRADIENT_AUDIT_SCHEMA_VERSION = "xunce-controlled-training-gradient-audit/v1"
CHECKPOINT_METADATA_SCHEMA_VERSION = "xunce-controlled-training-checkpoint-metadata/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-controlled-training-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-controlled-training-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_controlled_training_candidate_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_controlled_training_candidate_v1"

SUMMARY_FILE = "xunce-controlled-training-candidate-summary.json"
MANIFEST_FILE = "xunce-controlled-training-candidate-manifest.json"
LOSS_AUDIT_FILE = "xunce-controlled-training-loss-audit.json"
GRADIENT_AUDIT_FILE = "xunce-controlled-training-gradient-audit.json"
CHECKPOINT_METADATA_FILE = "xunce-controlled-training-checkpoint-metadata.json"
BOUNDARY_AUDIT_FILE = "xunce-controlled-training-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-controlled-training-rejection-report.json"
REPORT_FILE = "xunce-controlled-training-candidate-report.md"
CHECKPOINT_FILE = "xunce-controlled-training-candidate.pt"

ARCHITECTURE = "xunce_full_network_v1"
PASS_NEXT_REQUIRED_CHANGE = "post_training_offline_evaluation"
FAIL_NEXT_REQUIRED_CHANGE = "fix_controlled_training_candidate"
FIX_PREFLIGHT_NEXT_REQUIRED_CHANGE = "fix_guarded_training_candidate_preflight"
BOUNDARY_NEXT_REQUIRED_CHANGE = "resolve_xunce_controlled_training_boundary_rejections"

SOURCE_PREFLIGHT_SUMMARY = "xunce-guarded-training-candidate-preflight-summary.json"
PREFLIGHT_PASS_NEXT_REQUIRED_CHANGE = "controlled_training_candidate"

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
    parser = argparse.ArgumentParser(description="Run Xunce Controlled Training Candidate v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_controlled_training_candidate(
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


def run_xunce_controlled_training_candidate(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    preflight = _load_json(resolve_path(Path(config["source_training_preflight_root"]), repo_root) / SOURCE_PREFLIGHT_SUMMARY)
    source_audit = _source_audit(preflight)
    boundary_audit = _boundary_audit(preflight)
    generated_at = utc_now()

    training_result: dict[str, Any] | None = None
    if source_audit["source_preflight_audit_passed"] and boundary_audit["source_boundary_audit_passed"]:
        training_result = _run_training(config=config, paths=paths)
    decision = _decision(source_audit, boundary_audit, training_result, min_loss_improvement=config["min_loss_improvement"])
    checkpoint_metadata = _checkpoint_metadata(generated_at, config, paths, training_result, decision)
    if training_result and decision["status"] == "passed" and config["write_research_checkpoint"]:
        torch.save(
            {
                "schema_version": "xunce-controlled-training-candidate-checkpoint/v1",
                "model_state_dict": training_result["model"].state_dict(),
                "metadata": checkpoint_metadata,
            },
            paths["checkpoint"],
        )

    loss_audit = _loss_audit(training_result, decision)
    gradient_audit = _gradient_audit(training_result, decision)
    rejection_report = {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "source_preflight_audit_passed": source_audit["source_preflight_audit_passed"],
        "source_boundary_audit_passed": boundary_audit["source_boundary_audit_passed"],
        "training_executed": bool(training_result),
    }
    summary = _summary(
        generated_at,
        config_path,
        output_root,
        paths,
        config,
        preflight,
        source_audit,
        boundary_audit,
        loss_audit,
        gradient_audit,
        checkpoint_metadata,
        decision,
        repo_root,
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(path) for key, path in paths.items()},
        "summary_status": summary["status"],
        "next_required_change": summary["next_required_change"],
    }
    write_json(paths["loss_audit"], loss_audit)
    write_json(paths["gradient_audit"], gradient_audit)
    write_json(paths["checkpoint_metadata"], checkpoint_metadata)
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
    string_fields = ("source_training_preflight_root",)
    for key in string_fields:
        if not isinstance(normalized.get(key), str) or not normalized[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    int_fields = (
        "seed",
        "candidate_feature_count",
        "edge_feature_count",
        "memory_feature_count",
        "context_feature_count",
        "missing_indicator_count",
        "hidden_dim",
        "message_passing_layers",
        "candidate_count",
        "training_step_limit",
    )
    for key in int_fields:
        normalized[key] = _int_value(normalized.get(key), key)
    for key in ("candidate_feature_count", "edge_feature_count", "memory_feature_count", "context_feature_count", "hidden_dim", "message_passing_layers", "candidate_count", "training_step_limit"):
        if normalized[key] <= 0:
            raise ConfigError(f"{key} must be positive")
    if normalized["missing_indicator_count"] < 0:
        raise ConfigError("missing_indicator_count must be non-negative")
    for key in ("learning_rate", "value_loss_weight", "min_loss_improvement", "max_gradient_norm"):
        normalized[key] = _float_value(normalized.get(key), key)
    if normalized["learning_rate"] <= 0.0:
        raise ConfigError("learning_rate must be positive")
    if normalized["value_loss_weight"] < 0.0:
        raise ConfigError("value_loss_weight must be non-negative")
    if normalized["min_loss_improvement"] < 0.0:
        raise ConfigError("min_loss_improvement must be non-negative")
    if normalized["max_gradient_norm"] <= 0.0:
        raise ConfigError("max_gradient_norm must be positive")
    normalized["write_research_checkpoint"] = bool(normalized.get("write_research_checkpoint", True))
    return normalized


def _source_audit(preflight: dict[str, Any] | None) -> dict[str, Any]:
    reason_codes: list[str] = []
    if not isinstance(preflight, dict):
        reason_codes.append("missing_training_preflight_summary")
    else:
        if preflight.get("status") != "passed":
            reason_codes.append("training_preflight_not_passed")
        if preflight.get("next_required_change") != PREFLIGHT_PASS_NEXT_REQUIRED_CHANGE:
            reason_codes.append("training_preflight_wrong_next_required_change")
        if preflight.get("controlled_training_candidate_authorized") is not True:
            reason_codes.append("controlled_training_candidate_not_authorized")
    return {
        "schema_version": "xunce-controlled-training-source-preflight-audit/v1",
        "source_status": preflight.get("status") if isinstance(preflight, dict) else None,
        "source_next_required_change": preflight.get("next_required_change") if isinstance(preflight, dict) else None,
        "controlled_training_candidate_authorized": bool(preflight.get("controlled_training_candidate_authorized", False)) if isinstance(preflight, dict) else False,
        "reason_codes": unique_sorted(reason_codes),
        "source_preflight_audit_passed": not reason_codes,
    }


def _boundary_audit(preflight: dict[str, Any] | None) -> dict[str, Any]:
    violations: list[str] = []
    observed: dict[str, bool] = {}
    if isinstance(preflight, dict):
        for field in BOUNDARY_FIELDS:
            value = bool(preflight.get(field, False))
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


def _run_training(*, config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = XunceFullNetworkV1(
        candidate_feature_count=config["candidate_feature_count"],
        edge_feature_count=config["edge_feature_count"],
        memory_feature_count=config["memory_feature_count"],
        context_feature_count=config["context_feature_count"],
        missing_indicator_count=config["missing_indicator_count"],
        hidden_dim=config["hidden_dim"],
        message_passing_layers=config["message_passing_layers"],
        dropout=0.0,
    )
    batch = _synthetic_batch(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_history: list[float] = []
    policy_loss_history: list[float] = []
    value_loss_history: list[float] = []
    gradient_norm_history: list[float] = []
    non_finite_gradient_count = 0
    target_action = torch.tensor([0], dtype=torch.long)
    value_target = torch.tensor([1.0], dtype=torch.float32)

    with torch.no_grad():
        initial_outputs = model(**batch)
        initial_policy_loss = F.cross_entropy(initial_outputs.masked_logits, target_action)
        initial_value_loss = F.mse_loss(initial_outputs.value, value_target)
        initial_loss = initial_policy_loss + config["value_loss_weight"] * initial_value_loss
        loss_history.append(float(initial_loss.item()))
        policy_loss_history.append(float(initial_policy_loss.item()))
        value_loss_history.append(float(initial_value_loss.item()))

    for _step in range(config["training_step_limit"]):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        policy_loss = F.cross_entropy(outputs.masked_logits, target_action)
        value_loss = F.mse_loss(outputs.value, value_target)
        loss = policy_loss + config["value_loss_weight"] * value_loss
        if not torch.isfinite(loss):
            loss_history.append(float("nan"))
            non_finite_gradient_count += 1
            break
        loss.backward()
        step_non_finite = 0
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                step_non_finite += 1
        non_finite_gradient_count += step_non_finite
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_gradient_norm"])
        gradient_norm_history.append(float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm))
        if step_non_finite:
            break
        optimizer.step()
        with torch.no_grad():
            measured_outputs = model(**batch)
            measured_policy_loss = F.cross_entropy(measured_outputs.masked_logits, target_action)
            measured_value_loss = F.mse_loss(measured_outputs.value, value_target)
            measured_loss = measured_policy_loss + config["value_loss_weight"] * measured_value_loss
            loss_history.append(float(measured_loss.item()))
            policy_loss_history.append(float(measured_policy_loss.item()))
            value_loss_history.append(float(measured_value_loss.item()))

    loss_before = loss_history[0]
    loss_after = loss_history[-1]
    return {
        "model": model,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_improvement": loss_before - loss_after,
        "loss_decreased": loss_after < loss_before,
        "loss_history": loss_history,
        "policy_loss_history": policy_loss_history,
        "value_loss_history": value_loss_history,
        "gradient_norm_history": gradient_norm_history,
        "finite_loss": all(math.isfinite(item) for item in loss_history),
        "non_finite_gradient_count": non_finite_gradient_count,
        "finite_gradient_count": max(0, len(gradient_norm_history) - non_finite_gradient_count),
        "max_observed_gradient_norm": max(gradient_norm_history) if gradient_norm_history else 0.0,
        "training_step_count": len(gradient_norm_history),
        "parameter_count": parameter_count(model),
        "checkpoint_path": str(paths["checkpoint"]),
    }


def _synthetic_batch(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    candidate_count = config["candidate_count"]
    candidate_total = candidate_count * config["candidate_feature_count"]
    candidate_features = torch.linspace(-0.8, 0.8, steps=candidate_total, dtype=torch.float32).reshape(
        1, candidate_count, config["candidate_feature_count"]
    )
    context_total = candidate_count * config["context_feature_count"]
    context_features = torch.linspace(0.25, -0.25, steps=context_total, dtype=torch.float32).reshape(
        1, candidate_count, config["context_feature_count"]
    )
    memory_features = torch.linspace(-0.3, 0.3, steps=config["memory_feature_count"], dtype=torch.float32).reshape(
        1, config["memory_feature_count"]
    )
    if config["missing_indicator_count"]:
        missing = torch.zeros((1, candidate_count, config["missing_indicator_count"]), dtype=torch.float32)
        missing[:, -1, :] = 1.0
    else:
        missing = torch.empty((1, candidate_count, 0), dtype=torch.float32)
    edges = [(index, index + 1) for index in range(candidate_count - 1)]
    if candidate_count > 2:
        edges.append((0, candidate_count - 1))
    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_values = []
    for edge_idx, (left, right) in enumerate(edges):
        span = float(abs(right - left)) / max(1.0, float(candidate_count - 1))
        edge_values.append([span, float(left) / candidate_count, float(right) / candidate_count, float(edge_idx + 1) / len(edges), 1.0])
    edge_features = torch.tensor(edge_values, dtype=torch.float32)
    if edge_features.shape[1] != config["edge_feature_count"]:
        base = edge_features
        if config["edge_feature_count"] < base.shape[1]:
            edge_features = base[:, : config["edge_feature_count"]]
        else:
            pad = torch.zeros((base.shape[0], config["edge_feature_count"] - base.shape[1]), dtype=torch.float32)
            edge_features = torch.cat((base, pad), dim=1)
    action_mask = torch.ones((1, candidate_count), dtype=torch.bool)
    return {
        "candidate_features": candidate_features,
        "edge_features": edge_features,
        "edge_index": edge_index,
        "memory_features": memory_features,
        "context_features": context_features,
        "action_mask": action_mask,
        "candidate_missing_indicators": missing,
    }


def _decision(
    source_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
    training_result: dict[str, Any] | None,
    *,
    min_loss_improvement: float,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    reason_codes.extend(source_audit["reason_codes"])
    if not boundary_audit["source_boundary_audit_passed"]:
        reason_codes.append("controlled_training_source_boundary_violation")
    if training_result:
        if not training_result["finite_loss"] or training_result["non_finite_gradient_count"]:
            reason_codes.append("non_finite_training_signal")
        if not training_result["loss_decreased"]:
            reason_codes.append("loss_not_decreased")
        if training_result["loss_improvement"] < min_loss_improvement:
            reason_codes.append("loss_improvement_too_small")
    reason_codes = unique_sorted(reason_codes)
    if not reason_codes:
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif "controlled_training_source_boundary_violation" in reason_codes:
        next_required_change = BOUNDARY_NEXT_REQUIRED_CHANGE
    elif any(code.startswith("missing_training_preflight") or code.startswith("training_preflight") or code.startswith("controlled_training_candidate_not_authorized") for code in reason_codes):
        next_required_change = FIX_PREFLIGHT_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FAIL_NEXT_REQUIRED_CHANGE
    return {"status": "passed" if not reason_codes else "failed", "reason_codes": reason_codes, "next_required_change": next_required_change}


def _loss_audit(training_result: dict[str, Any] | None, decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": LOSS_AUDIT_SCHEMA_VERSION,
        "status": decision["status"],
        "loss_before": training_result["loss_before"] if training_result else None,
        "loss_after": training_result["loss_after"] if training_result else None,
        "loss_improvement": training_result["loss_improvement"] if training_result else None,
        "loss_decreased": bool(training_result and training_result["loss_decreased"]),
        "loss_history": training_result["loss_history"] if training_result else [],
        "policy_loss_history": training_result["policy_loss_history"] if training_result else [],
        "value_loss_history": training_result["value_loss_history"] if training_result else [],
    }


def _gradient_audit(training_result: dict[str, Any] | None, decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": GRADIENT_AUDIT_SCHEMA_VERSION,
        "status": decision["status"],
        "finite_gradient_count": training_result["finite_gradient_count"] if training_result else 0,
        "non_finite_gradient_count": training_result["non_finite_gradient_count"] if training_result else 0,
        "max_observed_gradient_norm": training_result["max_observed_gradient_norm"] if training_result else 0.0,
        "gradient_norm_history": training_result["gradient_norm_history"] if training_result else [],
    }


def _checkpoint_metadata(generated_at: str, config: dict[str, Any], paths: dict[str, Path], training_result: dict[str, Any] | None, decision: dict[str, Any]) -> dict[str, Any]:
    writes = bool(training_result and decision["status"] == "passed" and config["write_research_checkpoint"])
    metadata = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA_VERSION,
        "generated_at": generated_at,
        "architecture": ARCHITECTURE,
        "research_checkpoint_path": str(paths["checkpoint"]) if writes else None,
        "writes_research_checkpoint": writes,
        "publishes_checkpoint": False,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "training_step_count": training_result["training_step_count"] if training_result else 0,
        "parameter_count": training_result["parameter_count"] if training_result else None,
    }
    if training_result:
        metadata.update(
            {
                "loss_before": training_result["loss_before"],
                "loss_after": training_result["loss_after"],
                "loss_improvement": training_result["loss_improvement"],
            }
        )
    return metadata


def _summary(
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    config: dict[str, Any],
    preflight: dict[str, Any] | None,
    source_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
    loss_audit: dict[str, Any],
    gradient_audit: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    passed = decision["status"] == "passed"
    training_executed = loss_audit["loss_before"] is not None
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_training_preflight_status": preflight.get("status") if isinstance(preflight, dict) else None,
        "source_training_preflight_next_required_change": preflight.get("next_required_change") if isinstance(preflight, dict) else None,
        "controlled_training_candidate_authorized": source_audit["controlled_training_candidate_authorized"],
        "architecture": ARCHITECTURE,
        "parameter_count": checkpoint_metadata["parameter_count"],
        "training_step_count": checkpoint_metadata["training_step_count"],
        "loss_before": loss_audit["loss_before"],
        "loss_after": loss_audit["loss_after"],
        "loss_improvement": loss_audit["loss_improvement"],
        "loss_decreased": loss_audit["loss_decreased"],
        "finite_gradient_count": gradient_audit["finite_gradient_count"],
        "non_finite_gradient_count": gradient_audit["non_finite_gradient_count"],
        "max_gradient_norm": gradient_audit["max_observed_gradient_norm"],
        "source_preflight_audit_passed": source_audit["source_preflight_audit_passed"],
        "boundary_audit_passed": boundary_audit["boundary_audit_passed"],
        "source_boundary_audit_passed": boundary_audit["source_boundary_audit_passed"],
        "controlled_training_candidate_passed": passed,
        "runs_controlled_training_update": training_executed,
        "writes_research_checkpoint": checkpoint_metadata["writes_research_checkpoint"],
        "research_checkpoint_path": checkpoint_metadata["research_checkpoint_path"],
        "checkpoint_metadata_path": str(paths["checkpoint_metadata"]),
        "ppo_update_executed": False,
        "next_required_change": decision["next_required_change"],
        "git_provenance": {"current": git_snapshot(repo_root)},
    }
    summary.update(_closed_boundary_fields())
    return summary


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "loss_audit": output_root / LOSS_AUDIT_FILE,
        "gradient_audit": output_root / GRADIENT_AUDIT_FILE,
        "checkpoint_metadata": output_root / CHECKPOINT_METADATA_FILE,
        "boundary_audit": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
        "checkpoint": output_root / CHECKPOINT_FILE,
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


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Controlled Training Candidate v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- loss_before: `{summary['loss_before']}`",
            f"- loss_after: `{summary['loss_after']}`",
            f"- writes_research_checkpoint: `{summary['writes_research_checkpoint']}`",
            f"- publishes_checkpoint: `{summary['publishes_checkpoint']}`",
            f"- runs_new_ppo_update: `{summary['runs_new_ppo_update']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            "",
            "This is a bounded research training candidate. It writes a research-only checkpoint when passing, but it does not publish or install that checkpoint.",
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
