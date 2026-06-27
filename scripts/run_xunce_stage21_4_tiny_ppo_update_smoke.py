from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_EXPLORER_SRC = SCRIPT_DIR.parent / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_xunce_high_fidelity_real_map_comparison import _load_xunce_checkpoint  # noqa: E402
from xunce_full_network_common import XunceFullNetworkV1, parameter_count  # noqa: E402
from xunce_continuous_theta_action import (
    CONTINUOUS_THETA_ACTION_SPACE,
    continuous_theta_torch_log_prob,
)


SYNTHETIC_CREDIT_BEHAVIOR_POLICY_ID = "synthetic_credit_mixture_policy/v1"


CONFIG_SCHEMA_VERSION = "xunce-stage21-4-tiny-ppo-update-smoke-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-4-tiny-ppo-update-smoke-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-4-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-4-manifest/v1"
CHECKPOINT_SCHEMA_VERSION = "xunce-stage21-4-experimental-checkpoint/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_4_tiny_ppo_update_smoke_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/"
    "outputs/path_feedback_batch_xunce_stage21_4_tiny_ppo_update_smoke_v1"
)

SUMMARY_FILE = "xunce-stage21-4-tiny-ppo-update-smoke-summary.json"
LOSS_AUDIT_FILE = "xunce-stage21-4-ppo-loss-audit.jsonl"
GRADIENT_AUDIT_FILE = "xunce-stage21-4-gradient-audit.json"
CHECKPOINT_AUDIT_FILE = "xunce-stage21-4-checkpoint-audit.json"
ROUTING_FILE = "xunce-stage21-4-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-4-report.md"
MANIFEST_FILE = "xunce-stage21-4-manifest.json"
CHECKPOINT_FILE = "experimental-xunce-stage21-4-tiny-ppo-candidate.pt"
CHECKPOINT_METADATA_FILE = "experimental-xunce-stage21-4-tiny-ppo-candidate-metadata.json"

ROUTE_BOUNDARY = "resolve_stage21_4_tiny_ppo_boundary_rejections"
ROUTE_INPUTS = "rerun_stage21_4_required_inputs"
ROUTE_BATCH = "repair_stage21_4_train_split_batch_contract"
ROUTE_NUMERICS = "repair_stage21_4_ppo_update_numerics"
ROUTE_CHECKPOINT = "repair_stage21_4_experimental_checkpoint_isolation"
ROUTE_STAGE21_5 = "implement_stage21_5_single_seed_ppo_pilot"

FORBIDDEN_TRUE_FIELDS = (
    "stage21_4_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)

REQUIRED_BATCH_KEYS = (
    "candidate_features",
    "edge_features",
    "edge_index",
    "memory_features",
    "context_features",
    "action_mask",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.4 tiny Xunce PPO update smoke.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_4_tiny_ppo_update_smoke(
            config_path=Path(args.config),
            output_root=Path(args.output_root),
            repo_root=Path(args.repo_root),
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "train_transition_count": summary["train_transition_count"],
                "runs_new_ppo_update": summary["runs_new_ppo_update"],
                "experimental_checkpoint": summary["experimental_checkpoint"],
                "checkpoint_reload_passed": summary["checkpoint_reload_passed"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage21_4_tiny_ppo_update_smoke(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config_path = _resolve_path(config_path, repo_root)
    config = _load_config(config_path, repo_root=repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    boundary_reasons = _boundary_rejections(config, output_root)
    input_reasons = _input_rejections(config)
    summary21_3: dict[str, Any] = {}
    lineage21_3: dict[str, Any] = {}
    stage21_1_summary: dict[str, Any] = {}
    batch_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    checkpoint_audit: dict[str, Any] = _empty_checkpoint_audit()
    gradient_audit: dict[str, Any] = _empty_gradient_audit()
    loss_audit_rows: list[dict[str, Any]] = []
    model_audit: dict[str, Any] = {}
    xunce_config: dict[str, int] = {}
    parameter_delta_l2 = None
    sample_count_too_low = False

    if not boundary_reasons and not input_reasons:
        stage21_3_root = Path(config["stage21_3_ppo_batch_validation_root"])
        summary21_3 = _read_json(stage21_3_root / "xunce-stage21-3-ppo-batch-validation-summary.json")
        lineage21_3 = _read_json(stage21_3_root / "xunce-stage21-3-lineage-audit.json")
        batch_rows = _read_jsonl(stage21_3_root / "xunce-stage21-3-ppo-trainable-batch.jsonl")
        train_rows = [row for row in batch_rows if row.get("stage21_3_split") == "train"]
        stage21_1_root = Path(str(summary21_3.get("stage21_1_collector_root", "")))
        if stage21_1_root:
            stage21_1_summary_path = stage21_1_root / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json"
            if stage21_1_summary_path.is_file():
                stage21_1_summary = _read_json(stage21_1_summary_path)

    batch_reasons = _batch_rejections(train_rows, summary21_3, lineage21_3, config)
    numeric_reasons: list[str] = []
    checkpoint_reasons: list[str] = []
    status = "passed"
    route = ROUTE_STAGE21_5
    if boundary_reasons:
        status = "failed"
        route = ROUTE_BOUNDARY
    elif input_reasons:
        status = "failed"
        route = ROUTE_INPUTS
    elif batch_reasons:
        status = "failed"
        route = ROUTE_BATCH
    else:
        high_fidelity_config = _read_json(Path(config["high_fidelity_config"]))
        if any(_row_uses_continuous_theta(row) for row in train_rows):
            init_seed = config.get("continuous_theta_head_init_seed", stage21_1_summary.get("sampling_seed"))
            if init_seed is not None:
                high_fidelity_config["continuous_theta_head_init_seed"] = int(init_seed)
        model_audit, model, xunce_config = _load_xunce_checkpoint(
            Path(config["xunce_candidate_checkpoint"]),
            config=high_fidelity_config,
            repo_root=repo_root,
        )
        if model is None or not model_audit.get("checkpoint_loaded"):
            checkpoint_reasons.append("source_xunce_checkpoint_not_loadable")
        else:
            sample_count_too_low = len(train_rows) < int(config["min_transition_count_for_performance_claim"])
            source_hash_before = _sha256_file(Path(config["xunce_candidate_checkpoint"]))
            model.train()
            initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
            try:
                sampling_temperature = _resolve_sampling_temperature(train_rows, stage21_1_summary, config)
                loss_audit_rows, gradient_audit = _run_tiny_ppo_update(
                    model,
                    train_rows,
                    optimizer=optimizer,
                    epochs=int(config["epochs"]),
                    clip_ratio=float(config["clip_ratio"]),
                    policy_loss_coefficient=float(config["policy_loss_coefficient"]),
                    value_loss_coefficient=float(config["value_loss_coefficient"]),
                    entropy_coefficient=float(config["entropy_coefficient"]),
                    advantage_clip_abs=float(config["advantage_clip_abs"]),
                    normalize_minibatch_advantages=bool(config["normalize_minibatch_advantages"]),
                    loss_scale=float(config["loss_scale"]),
                    max_grad_norm=float(config["max_grad_norm"]),
                    sampling_temperature=sampling_temperature,
                )
            except (ValueError, RuntimeError) as exc:
                numeric_reasons.append(str(exc) or "stage21_4_ppo_update_exception")
            if not numeric_reasons:
                parameter_delta_l2 = _parameter_delta_l2(model, initial_state)
                if not _finite(parameter_delta_l2) or float(parameter_delta_l2) <= 0.0:
                    numeric_reasons.append("parameter_delta_not_positive")
                if not _gradient_audit_passed(gradient_audit):
                    numeric_reasons.append("gradient_audit_failed")
                if any(not _loss_row_finite(row) for row in loss_audit_rows):
                    numeric_reasons.append("non_finite_loss_audit")
                if any(abs(float(row["post_update_approx_kl"])) > float(config["max_abs_approx_kl"]) for row in loss_audit_rows):
                    numeric_reasons.append("post_update_approx_kl_exceeded")
            if not numeric_reasons:
                checkpoint_path = output_root / CHECKPOINT_FILE
                metadata_path = output_root / CHECKPOINT_METADATA_FILE
                checkpoint_audit = _write_and_reload_checkpoint(
                    model,
                    checkpoint_path=checkpoint_path,
                    metadata_path=metadata_path,
                    xunce_config=xunce_config,
                    config=config,
                    repo_root=repo_root,
                    source_checkpoint_hash=source_hash_before,
                    stage21_3_summary=summary21_3,
                    parameter_delta_l2=float(parameter_delta_l2),
                )
                source_hash_after = _sha256_file(Path(config["xunce_candidate_checkpoint"]))
                if source_hash_before != source_hash_after:
                    checkpoint_reasons.append("source_checkpoint_hash_changed")
                if not checkpoint_audit.get("checkpoint_reload_passed"):
                    checkpoint_reasons.append("experimental_checkpoint_reload_failed")
                if config["require_d_drive_output_root"] and checkpoint_path.drive.upper() != "D:":
                    checkpoint_reasons.append("experimental_checkpoint_not_on_d_drive")

        if numeric_reasons:
            status = "failed"
            route = ROUTE_NUMERICS
        elif checkpoint_reasons:
            status = "failed"
            route = ROUTE_CHECKPOINT

    blocking = list(boundary_reasons + input_reasons + batch_reasons + numeric_reasons + checkpoint_reasons)
    return _write_outputs(
        config=config,
        output_root=output_root,
        config_path=config_path,
        summary21_3=summary21_3,
        lineage21_3=lineage21_3,
        model_audit=model_audit,
        xunce_config=xunce_config,
        train_rows=train_rows,
        loss_audit_rows=loss_audit_rows,
        gradient_audit=gradient_audit,
        checkpoint_audit=checkpoint_audit,
        sample_count_too_low=sample_count_too_low,
        parameter_delta_l2=parameter_delta_l2,
        blocking_reason_codes=blocking,
        status=status,
        route=route,
    )


def _run_tiny_ppo_update(
    model: XunceFullNetworkV1,
    rows: list[dict[str, Any]],
    *,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    clip_ratio: float,
    policy_loss_coefficient: float,
    value_loss_coefficient: float,
    entropy_coefficient: float,
    advantage_clip_abs: float,
    normalize_minibatch_advantages: bool,
    loss_scale: float,
    max_grad_norm: float,
    sampling_temperature: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    gradient_audit: dict[str, Any] = _empty_gradient_audit()
    for epoch_index in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        policy_losses: list[torch.Tensor] = []
        value_losses: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        old_log_probs: list[torch.Tensor] = []
        new_log_probs: list[torch.Tensor] = []
        ratios: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        returns: list[torch.Tensor] = []
        advantages: list[torch.Tensor] = []
        raw_advantages = [_required_float(row.get("advantage"), "advantage") for row in rows]
        effective_advantages, advantage_stats = _effective_advantages(
            raw_advantages,
            normalize_minibatch_advantages=normalize_minibatch_advantages,
            advantage_clip_abs=advantage_clip_abs,
        )
        for row, advantage in zip(rows, effective_advantages):
            tensors = _deserialize_xunce_batch(row.get("xunce_batch"))
            action_index = _required_int(row.get("action_index"), "action_index")
            old_log_prob = _required_float(row.get("old_log_prob"), "old_log_prob")
            ret = _required_float(row.get("return"), "return")
            sampling_mask = _sampling_mask(row, tensors["action_mask"], action_index)
            output = model(**tensors)
            logits = output.masked_logits[0] / float(sampling_temperature)
            logits = logits.masked_fill(~sampling_mask, -1.0e9)
            if _row_uses_continuous_theta(row):
                theta_detail = continuous_theta_torch_log_prob(
                    point_logits=logits,
                    theta_mu_rad=output.theta_mu_rad[0],
                    theta_kappa=output.theta_kappa[0],
                    action_index=action_index,
                    theta_rad=_required_float(row.get("selected_theta_rad"), "selected_theta_rad"),
                )
                new_log_prob = theta_detail["total_log_prob"]
                entropy_value = theta_detail["total_entropy"]
            else:
                distribution = torch.distributions.Categorical(logits=logits)
                action = torch.tensor(action_index, dtype=torch.long)
                new_log_prob = distribution.log_prob(action)
                entropy_value = distribution.entropy()
            old_log_prob_tensor = torch.tensor(old_log_prob, dtype=torch.float32)
            advantage_tensor = torch.tensor(advantage, dtype=torch.float32)
            return_tensor = torch.tensor(ret, dtype=torch.float32)
            ratio = torch.exp(new_log_prob - old_log_prob_tensor)
            unclipped = ratio * advantage_tensor
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage_tensor
            policy_losses.append(-torch.min(unclipped, clipped))
            value_losses.append(F.mse_loss(output.value[0], return_tensor))
            entropies.append(entropy_value)
            old_log_probs.append(old_log_prob_tensor)
            new_log_probs.append(new_log_prob)
            ratios.append(ratio)
            values.append(output.value[0])
            returns.append(return_tensor)
            advantages.append(advantage_tensor)

        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean()
        policy_loss_component = loss_scale * policy_loss_coefficient * policy_loss
        value_loss_component = loss_scale * value_loss_coefficient * value_loss
        entropy_loss_component = loss_scale * (-entropy_coefficient * entropy)
        total_loss = policy_loss_component + value_loss_component + entropy_loss_component
        if not torch.isfinite(total_loss):
            raise ValueError("non_finite_total_loss")
        component_grad_norms = _component_grad_norms(
            model,
            {
                "total_loss_grad_norm": total_loss,
                "policy_loss_grad_norm": policy_loss_component,
                "value_loss_grad_norm": value_loss_component,
                "entropy_loss_grad_norm": entropy_loss_component,
                "unscaled_policy_loss_grad_norm": policy_loss,
                "unscaled_value_loss_grad_norm": value_loss,
            },
        )
        total_loss.backward()
        gradient_audit = _gradient_audit(model)
        gradient_audit["component_grad_norms"] = component_grad_norms
        gradient_audit["loss_scale"] = float(loss_scale)
        gradient_audit["policy_loss_coefficient"] = float(policy_loss_coefficient)
        gradient_audit["advantage_clip_abs"] = float(advantage_clip_abs)
        gradient_audit["normalize_minibatch_advantages"] = bool(normalize_minibatch_advantages)
        if max_grad_norm > 0:
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            gradient_audit["pre_clip_grad_norm"] = float(pre_clip_norm)
            gradient_audit["post_clip_grad_norm"] = _current_grad_norm(model)
            gradient_audit["max_grad_norm"] = float(max_grad_norm)
        optimizer.step()

        post_metrics = _evaluate_rows(
            model,
            rows,
            clip_ratio=clip_ratio,
            sampling_temperature=sampling_temperature,
        )
        ratio_tensor = torch.stack(ratios).detach()
        approx_kl = (torch.stack(old_log_probs).detach() - torch.stack(new_log_probs).detach()).mean()
        audit_rows.append(
            {
                "schema_version": "xunce-stage21-4-ppo-loss-audit/v1",
                "epoch_index": epoch_index,
                "transition_count": len(rows),
                "total_loss": float(total_loss.detach()),
                "unscaled_total_loss": float(
                    (policy_loss_coefficient * policy_loss + value_loss_coefficient * value_loss - entropy_coefficient * entropy).detach()
                ),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "policy_loss_component": float(policy_loss_component.detach()),
                "value_loss_component": float(value_loss_component.detach()),
                "entropy_loss_component": float(entropy_loss_component.detach()),
                "loss_scale": float(loss_scale),
                "policy_loss_coefficient": float(policy_loss_coefficient),
                "value_loss_coefficient": float(value_loss_coefficient),
                "entropy_coefficient": float(entropy_coefficient),
                "advantage_clip_abs": float(advantage_clip_abs),
                "normalize_minibatch_advantages": bool(normalize_minibatch_advantages),
                **advantage_stats,
                **component_grad_norms,
                "pre_update_approx_kl": float(approx_kl.detach()),
                "post_update_approx_kl": post_metrics["approx_kl"],
                "clip_fraction": float((torch.abs(ratio_tensor - 1.0) > clip_ratio).to(torch.float32).mean()),
                "post_update_clip_fraction": post_metrics["clip_fraction"],
                "ratio_min": float(ratio_tensor.min()),
                "ratio_max": float(ratio_tensor.max()),
                "value_mean": float(torch.stack(values).detach().mean()),
                "return_mean": float(torch.stack(returns).detach().mean()),
                "advantage_mean": float(torch.stack(advantages).detach().mean()),
            }
        )
    return audit_rows, gradient_audit


def _effective_advantages(
    raw_advantages: list[float],
    *,
    normalize_minibatch_advantages: bool,
    advantage_clip_abs: float,
) -> tuple[list[float], dict[str, Any]]:
    effective = [float(value) for value in raw_advantages]
    raw_mean = sum(effective) / len(effective) if effective else 0.0
    raw_variance = sum((value - raw_mean) ** 2 for value in effective) / len(effective) if effective else 0.0
    raw_std = raw_variance ** 0.5
    normalization_applied = normalize_minibatch_advantages and len(effective) >= 2 and raw_std > 1.0e-12
    if normalization_applied:
        effective = [(value - raw_mean) / raw_std for value in effective]
    clipped_count = 0
    if advantage_clip_abs > 0.0:
        clipped = []
        for value in effective:
            bounded = max(-advantage_clip_abs, min(advantage_clip_abs, value))
            if bounded != value:
                clipped_count += 1
            clipped.append(bounded)
        effective = clipped
    eff_mean = sum(effective) / len(effective) if effective else 0.0
    eff_variance = sum((value - eff_mean) ** 2 for value in effective) / len(effective) if effective else 0.0
    eff_std = eff_variance ** 0.5
    return effective, {
        "raw_advantage_mean": raw_mean,
        "raw_advantage_std": raw_std,
        "raw_advantage_min": min(raw_advantages) if raw_advantages else None,
        "raw_advantage_max": max(raw_advantages) if raw_advantages else None,
        "effective_advantage_mean": eff_mean,
        "effective_advantage_std": eff_std,
        "effective_advantage_min": min(effective) if effective else None,
        "effective_advantage_max": max(effective) if effective else None,
        "minibatch_advantage_normalization_applied": normalization_applied,
        "advantage_clipped_count": clipped_count,
        "advantage_clipped_fraction": clipped_count / len(effective) if effective else 0.0,
    }


def _component_grad_norms(model: XunceFullNetworkV1, components: dict[str, torch.Tensor]) -> dict[str, float | None]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    result: dict[str, float | None] = {}
    for name, component in components.items():
        if not torch.isfinite(component):
            result[name] = None
            continue
        grads = torch.autograd.grad(component, params, retain_graph=True, allow_unused=True)
        total_sq = 0.0
        finite = True
        for grad in grads:
            if grad is None:
                continue
            if not torch.isfinite(grad).all():
                finite = False
                break
            total_sq += float(torch.sum(grad.detach() ** 2))
        result[name] = math.sqrt(total_sq) if finite else None
    return result


def _evaluate_rows(
    model: XunceFullNetworkV1,
    rows: list[dict[str, Any]],
    *,
    clip_ratio: float,
    sampling_temperature: float,
) -> dict[str, float]:
    model.eval()
    old_values: list[float] = []
    new_values: list[float] = []
    ratios: list[float] = []
    with torch.no_grad():
        for row in rows:
            tensors = _deserialize_xunce_batch(row.get("xunce_batch"))
            action_index = _required_int(row.get("action_index"), "action_index")
            sampling_mask = _sampling_mask(row, tensors["action_mask"], action_index)
            output = model(**tensors)
            logits = output.masked_logits[0] / float(sampling_temperature)
            logits = logits.masked_fill(~sampling_mask, -1.0e9)
            if _row_uses_continuous_theta(row):
                theta_detail = continuous_theta_torch_log_prob(
                    point_logits=logits,
                    theta_mu_rad=output.theta_mu_rad[0],
                    theta_kappa=output.theta_kappa[0],
                    action_index=action_index,
                    theta_rad=_required_float(row.get("selected_theta_rad"), "selected_theta_rad"),
                )
                new_log_prob = float(theta_detail["total_log_prob"])
            else:
                distribution = torch.distributions.Categorical(logits=logits)
                new_log_prob = float(distribution.log_prob(torch.tensor(action_index, dtype=torch.long)))
            old_log_prob = _required_float(row.get("old_log_prob"), "old_log_prob")
            old_values.append(old_log_prob)
            new_values.append(new_log_prob)
            ratios.append(float(math.exp(new_log_prob - old_log_prob)))
    model.train()
    if not rows:
        return {"approx_kl": float("nan"), "clip_fraction": float("nan")}
    return {
        "approx_kl": float(sum(old - new for old, new in zip(old_values, new_values)) / len(rows)),
        "clip_fraction": float(sum(1 for ratio in ratios if abs(ratio - 1.0) > clip_ratio) / len(ratios)),
    }


def _row_uses_continuous_theta(row: dict[str, Any]) -> bool:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    return (row.get("action_space_type") or info.get("action_space_type")) == CONTINUOUS_THETA_ACTION_SPACE


def _row_uses_synthetic_credit_behavior_policy(row: dict[str, Any]) -> bool:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    return (row.get("behavior_policy_id") or info.get("behavior_policy_id")) == SYNTHETIC_CREDIT_BEHAVIOR_POLICY_ID


def _deserialize_xunce_batch(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("xunce_batch_missing")
    tensors: dict[str, torch.Tensor] = {}
    for key in REQUIRED_BATCH_KEYS:
        if key not in payload:
            raise ValueError(f"xunce_batch_missing_{key}")
        tensors[key] = _tensor_from_payload(payload[key], key)
    if "candidate_missing_indicators" in payload:
        tensors["candidate_missing_indicators"] = _tensor_from_payload(payload["candidate_missing_indicators"], "candidate_missing_indicators")
    return tensors


def _tensor_from_payload(payload: Any, key: str) -> torch.Tensor:
    if not isinstance(payload, dict):
        raise ValueError(f"{key}_tensor_payload_invalid")
    dtype_name = str(payload.get("dtype", "float32"))
    dtype = {
        "bool": torch.bool,
        "int64": torch.long,
        "long": torch.long,
        "float32": torch.float32,
        "float": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"{key}_tensor_dtype_unsupported")
    tensor = torch.tensor(payload.get("values"), dtype=dtype)
    expected_shape = payload.get("shape")
    if isinstance(expected_shape, list) and list(tensor.shape) != [int(item) for item in expected_shape]:
        raise ValueError(f"{key}_tensor_shape_mismatch")
    return tensor


def _sampling_mask(row: dict[str, Any], action_mask: torch.Tensor, action_index: int) -> torch.Tensor:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    raw_mask = info.get("sampling_mask")
    if raw_mask is None:
        raise ValueError("sampling_mask_missing")
    mask = torch.tensor([bool(value) for value in raw_mask], dtype=torch.bool)
    if mask.ndim != 1 or mask.shape[0] != int(action_mask.shape[1]):
        raise ValueError("sampling_mask_shape_mismatch")
    info_action_mask = _required_mask(info.get("action_mask"), "action_mask", mask.shape[0])
    hard_risk_clean_mask = _required_mask(info.get("hard_risk_clean_mask"), "hard_risk_clean_mask", mask.shape[0])
    batch_action_mask = action_mask[0].detach().cpu().bool()
    if not torch.equal(info_action_mask, batch_action_mask):
        raise ValueError("action_mask_mismatch_between_info_and_xunce_batch")
    if bool((mask & ~info_action_mask).any()):
        raise ValueError("sampling_mask_not_subset_of_action_mask")
    if bool((mask & ~hard_risk_clean_mask).any()):
        raise ValueError("sampling_mask_not_subset_of_hard_risk_clean_mask")
    if action_index < 0 or action_index >= mask.shape[0] or not bool(mask[action_index]):
        raise ValueError("action_not_allowed_by_sampling_mask")
    if not bool(mask.any()):
        raise ValueError("sampling_mask_has_no_valid_action")
    return mask


def _required_mask(value: Any, name: str, expected_count: int) -> torch.Tensor:
    if not isinstance(value, list):
        raise ValueError(f"{name}_missing")
    mask = torch.tensor([bool(item) for item in value], dtype=torch.bool)
    if mask.ndim != 1 or mask.shape[0] != expected_count:
        raise ValueError(f"{name}_shape_mismatch")
    return mask


def _resolve_sampling_temperature(rows: list[dict[str, Any]], stage21_1_summary: dict[str, Any], config: dict[str, Any]) -> float:
    summary_temperature = _finite(stage21_1_summary.get("sampling_temperature"))
    config_temperature = _finite(config.get("sampling_temperature"))
    if summary_temperature is None:
        raise ValueError("stage21_1_sampling_temperature_missing")
    if config_temperature is not None and abs(float(config_temperature) - float(summary_temperature)) > 1.0e-9:
        raise ValueError("stage21_4_config_sampling_temperature_mismatch")
    row_temperatures: list[float] = []
    for row in rows:
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        value = info.get("sampling_temperature")
        if value is not None:
            row_temperature = _finite(value)
            if row_temperature is None:
                raise ValueError("row_sampling_temperature_non_finite")
            row_temperatures.append(float(row_temperature))
    if row_temperatures and any(abs(value - float(summary_temperature)) > 1.0e-9 for value in row_temperatures):
        raise ValueError("row_sampling_temperature_mismatch")
    return float(summary_temperature)


def _write_and_reload_checkpoint(
    model: XunceFullNetworkV1,
    *,
    checkpoint_path: Path,
    metadata_path: Path,
    xunce_config: dict[str, int],
    config: dict[str, Any],
    repo_root: Path,
    source_checkpoint_hash: str | None,
    stage21_3_summary: dict[str, Any],
    parameter_delta_l2: float,
) -> dict[str, Any]:
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture": "xunce_full_network_v1",
        "experimental_only": True,
        "stage": "21.4",
        "stage_id": "xunce-stage21-4-tiny-ppo-update-smoke",
        "source_checkpoint_path": str(config["xunce_candidate_checkpoint"]),
        "source_checkpoint_sha256": source_checkpoint_hash,
        "stage21_3_ppo_batch_validation_root": str(config["stage21_3_ppo_batch_validation_root"]),
        "stage21_3_reward_profile_hash": stage21_3_summary.get("reward_profile_hash"),
        "parameter_delta_l2": parameter_delta_l2,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "training_or_release_authorized": False,
        "canary_traffic_fraction": 0.0,
        "runs_new_ppo_update": True,
        **{key: int(value) for key, value in xunce_config.items()},
    }
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "metadata": metadata,
    }
    torch.save(payload, checkpoint_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    high_fidelity_config = _read_json(Path(config["high_fidelity_config"]))
    reload_audit, reloaded_model, reloaded_config = _load_xunce_checkpoint(
        checkpoint_path,
        config=high_fidelity_config,
        repo_root=repo_root,
    )
    checkpoint_hash = _sha256_file(checkpoint_path)
    return {
        "schema_version": "xunce-stage21-4-checkpoint-audit/v1",
        "experimental_checkpoint_path": str(checkpoint_path),
        "experimental_checkpoint_metadata": str(metadata_path),
        "experimental_checkpoint_exists": checkpoint_path.is_file(),
        "experimental_checkpoint_size_bytes": checkpoint_path.stat().st_size if checkpoint_path.is_file() else 0,
        "experimental_checkpoint_sha256": checkpoint_hash,
        "checkpoint_reload_passed": bool(reloaded_model is not None and reload_audit.get("checkpoint_loaded")),
        "reload_audit": reload_audit,
        "reloaded_model_config": reloaded_config,
        "metadata": metadata,
        "parameter_count": parameter_count(model),
    }


def _write_outputs(
    *,
    config: dict[str, Any],
    output_root: Path,
    config_path: Path,
    summary21_3: dict[str, Any],
    lineage21_3: dict[str, Any],
    model_audit: dict[str, Any],
    xunce_config: dict[str, int],
    train_rows: list[dict[str, Any]],
    loss_audit_rows: list[dict[str, Any]],
    gradient_audit: dict[str, Any],
    checkpoint_audit: dict[str, Any],
    sample_count_too_low: bool,
    parameter_delta_l2: float | None,
    blocking_reason_codes: list[str],
    status: str,
    route: str,
) -> dict[str, Any]:
    paths = {
        "summary": output_root / SUMMARY_FILE,
        "loss_audit": output_root / LOSS_AUDIT_FILE,
        "gradient_audit": output_root / GRADIENT_AUDIT_FILE,
        "checkpoint_audit": output_root / CHECKPOINT_AUDIT_FILE,
        "routing": output_root / ROUTING_FILE,
        "report": output_root / REPORT_FILE,
        "manifest": output_root / MANIFEST_FILE,
    }
    _write_jsonl(paths["loss_audit"], loss_audit_rows)
    paths["gradient_audit"].write_text(json.dumps(gradient_audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    paths["checkpoint_audit"].write_text(json.dumps(checkpoint_audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "reason_codes": _unique_sorted(blocking_reason_codes),
        "stage19_authorized": False,
        "stage21_4_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": bool(config["runs_new_ppo_update"] and status == "passed"),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    paths["routing"].write_text(json.dumps(routing, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "next_required_change": route,
        "reason_codes": _unique_sorted(blocking_reason_codes),
        "blocking_reason_codes": _unique_sorted(blocking_reason_codes),
        "stage21_3_status": summary21_3.get("status"),
        "stage21_3_next_required_change": summary21_3.get("next_required_change"),
        "stage21_3_ppo_batch_validation_root": str(config["stage21_3_ppo_batch_validation_root"]),
        "source_xunce_candidate_checkpoint": str(config["xunce_candidate_checkpoint"]),
        "source_xunce_checkpoint_sha256": model_audit.get("checkpoint_sha256"),
        "source_checkpoint_read_only": True,
        "train_transition_count": len(train_rows),
        "sample_count_too_low_for_performance_claim": bool(sample_count_too_low),
        "epochs": int(config["epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "clip_ratio": float(config["clip_ratio"]),
        "policy_loss_coefficient": float(config["policy_loss_coefficient"]),
        "value_loss_coefficient": float(config["value_loss_coefficient"]),
        "entropy_coefficient": float(config["entropy_coefficient"]),
        "advantage_clip_abs": float(config["advantage_clip_abs"]),
        "normalize_minibatch_advantages": bool(config["normalize_minibatch_advantages"]),
        "loss_scale": float(config["loss_scale"]),
        "parameter_delta_l2": parameter_delta_l2,
        "gradient_finite": bool(gradient_audit.get("grad_norm_finite")),
        "pre_clip_grad_norm": gradient_audit.get("pre_clip_grad_norm"),
        "post_clip_grad_norm": gradient_audit.get("post_clip_grad_norm"),
        "total_loss_grad_norm": gradient_audit.get("component_grad_norms", {}).get("total_loss_grad_norm"),
        "policy_loss_grad_norm": gradient_audit.get("component_grad_norms", {}).get("policy_loss_grad_norm"),
        "value_loss_grad_norm": gradient_audit.get("component_grad_norms", {}).get("value_loss_grad_norm"),
        "entropy_loss_grad_norm": gradient_audit.get("component_grad_norms", {}).get("entropy_loss_grad_norm"),
        "gradient_parameter_count": gradient_audit.get("parameter_with_grad_count", 0),
        "loss_audit_row_count": len(loss_audit_rows),
        "synthetic_credit_behavior_policy_row_count": sum(
            1 for row in train_rows if _row_uses_synthetic_credit_behavior_policy(row)
        ),
        "old_log_prob_source": "behavior_policy_when_present",
        "final_total_loss": loss_audit_rows[-1]["total_loss"] if loss_audit_rows else None,
        "final_policy_loss": loss_audit_rows[-1]["policy_loss"] if loss_audit_rows else None,
        "final_value_loss": loss_audit_rows[-1]["value_loss"] if loss_audit_rows else None,
        "final_entropy": loss_audit_rows[-1]["entropy"] if loss_audit_rows else None,
        "final_post_update_approx_kl": loss_audit_rows[-1]["post_update_approx_kl"] if loss_audit_rows else None,
        "checkpoint_reload_passed": bool(checkpoint_audit.get("checkpoint_reload_passed")),
        "experimental_checkpoint": bool(checkpoint_audit.get("experimental_checkpoint_exists")),
        "experimental_checkpoint_path": checkpoint_audit.get("experimental_checkpoint_path"),
        "experimental_checkpoint_sha256": checkpoint_audit.get("experimental_checkpoint_sha256"),
        "xunce_model_config": xunce_config,
        "reward_profile_hash": summary21_3.get("reward_profile_hash") or lineage21_3.get("reward_profile_hash"),
        "summary": str(paths["summary"]),
        "loss_audit": str(paths["loss_audit"]),
        "gradient_audit": str(paths["gradient_audit"]),
        "checkpoint_audit": str(paths["checkpoint_audit"]),
        "routing": str(paths["routing"]),
        "report": str(paths["report"]),
        "manifest": str(paths["manifest"]),
        "stage21_4_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": bool(config["runs_new_ppo_update"] and status == "passed"),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    paths["report"].write_text(_report_markdown(summary), encoding="utf-8")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": summary["generated_at"],
        "config": str(config_path),
        "summary_status": status,
        "next_required_change": route,
        "artifacts": {key: str(value) for key, value in paths.items()},
        "experimental_checkpoint": summary["experimental_checkpoint_path"],
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _report_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 21.4 Tiny PPO Update Smoke",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- train_transition_count: `{summary['train_transition_count']}`",
            f"- sample_count_too_low_for_performance_claim: `{summary['sample_count_too_low_for_performance_claim']}`",
            f"- final_total_loss: `{summary['final_total_loss']}`",
            f"- final_post_update_approx_kl: `{summary['final_post_update_approx_kl']}`",
            f"- parameter_delta_l2: `{summary['parameter_delta_l2']}`",
            f"- checkpoint_reload_passed: `{summary['checkpoint_reload_passed']}`",
            "",
            "This stage performs an offline experimental PPO smoke update only. It does not publish a checkpoint, replace the default policy, connect a real executor, or start canary traffic.",
            "",
        ]
    )


def _batch_rejections(
    train_rows: list[dict[str, Any]],
    summary21_3: dict[str, Any],
    lineage21_3: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if summary21_3.get("status") != "passed":
        reasons.append("stage21_3_status_not_passed")
    if summary21_3.get("next_required_change") != "implement_stage21_4_tiny_ppo_update_smoke":
        reasons.append("stage21_3_route_not_stage21_4")
    if lineage21_3.get("reward_profile_hash_count") not in (None, 1):
        reasons.append("stage21_3_reward_profile_hash_count_not_one")
    if len(train_rows) < int(config["min_train_transition_count"]):
        reasons.append("stage21_4_train_transition_count_short")
    for row in train_rows:
        if row.get("stage21_3_split") != "train":
            reasons.append("non_train_split_row_selected")
        if not row.get("transition_trainable", True) or not row.get("reward_trainable", True):
            reasons.append("non_trainable_row_selected")
        if _finite(row.get("old_log_prob")) is None:
            reasons.append("non_finite_old_log_prob")
        if _row_uses_continuous_theta(row):
            if _finite(row.get("selected_theta_rad")) is None:
                reasons.append("continuous_theta_selected_theta_rad_missing")
            if _finite(row.get("old_point_log_prob")) is None:
                reasons.append("continuous_theta_old_point_log_prob_missing")
            if _finite(row.get("old_theta_log_prob")) is None:
                reasons.append("continuous_theta_old_theta_log_prob_missing")
            old_total = _finite(row.get("old_log_prob"))
            old_point = _finite(row.get("old_point_log_prob"))
            old_theta = _finite(row.get("old_theta_log_prob"))
            if (
                old_total is not None
                and old_point is not None
                and old_theta is not None
                and abs(old_total - (old_point + old_theta)) > 1.0e-5
            ):
                reasons.append("continuous_theta_old_log_prob_decomposition_mismatch")
        if _row_uses_synthetic_credit_behavior_policy(row):
            old_total = _finite(row.get("old_log_prob"))
            old_behavior_total = _finite(row.get("old_behavior_log_prob"))
            if old_total is None or old_behavior_total is None:
                reasons.append("synthetic_credit_behavior_logprob_missing")
            elif abs(old_total - old_behavior_total) > 1.0e-5:
                reasons.append("synthetic_credit_behavior_logprob_mismatch")
        if _finite(row.get("return")) is None:
            reasons.append("non_finite_return")
        if _finite(row.get("advantage")) is None:
            reasons.append("non_finite_advantage")
        if not isinstance(row.get("xunce_batch"), dict):
            reasons.append("xunce_batch_missing")
    return _unique_sorted(reasons)


def _boundary_rejections(config: dict[str, Any], output_root: Path) -> list[str]:
    reasons: list[str] = []
    for field in FORBIDDEN_TRUE_FIELDS:
        if config.get(field) is True:
            reasons.append(field)
    if config.get("offline_ppo_update_smoke_authorized") is not True:
        reasons.append("offline_ppo_update_smoke_not_authorized")
    if config.get("runs_new_ppo_update") is not True:
        reasons.append("runs_new_ppo_update_not_enabled_for_stage21_4_smoke")
    if float(config.get("canary_traffic_fraction", 0.0)) > 0.0:
        reasons.append("canary_traffic_fraction")
    if config.get("require_d_drive_output_root") and output_root.drive.upper() != "D:":
        reasons.append("output_root_not_on_d_drive")
    return reasons


def _input_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    stage21_3_root = Path(config["stage21_3_ppo_batch_validation_root"])
    for name in (
        "xunce-stage21-3-ppo-batch-validation-summary.json",
        "xunce-stage21-3-lineage-audit.json",
        "xunce-stage21-3-ppo-trainable-batch.jsonl",
    ):
        if not (stage21_3_root / name).is_file():
            reasons.append(f"missing_{name}")
    if not Path(config["xunce_candidate_checkpoint"]).is_file():
        reasons.append("missing_xunce_candidate_checkpoint")
    if not Path(config["high_fidelity_config"]).is_file():
        reasons.append("missing_high_fidelity_config")
    return reasons


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_config(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    path_fields = (
        "stage21_3_ppo_batch_validation_root",
        "xunce_candidate_checkpoint",
        "high_fidelity_config",
    )
    for field in path_fields:
        if field not in config:
            raise ConfigError(f"{field} is required")
        config[field] = str(_resolve_path(Path(config[field]), repo_root))
    config["epochs"] = _positive_int(config.get("epochs", 1), "epochs")
    config["min_train_transition_count"] = _nonnegative_int(config.get("min_train_transition_count", 1), "min_train_transition_count")
    config["min_transition_count_for_performance_claim"] = _nonnegative_int(
        config.get("min_transition_count_for_performance_claim", 200),
        "min_transition_count_for_performance_claim",
    )
    config["learning_rate"] = _positive_float(config.get("learning_rate", 1.0e-5), "learning_rate")
    config["clip_ratio"] = _positive_float(config.get("clip_ratio", 0.2), "clip_ratio")
    config["policy_loss_coefficient"] = _nonnegative_float(config.get("policy_loss_coefficient", 1.0), "policy_loss_coefficient")
    config["value_loss_coefficient"] = _nonnegative_float(config.get("value_loss_coefficient", 0.5), "value_loss_coefficient")
    config["entropy_coefficient"] = _nonnegative_float(config.get("entropy_coefficient", 0.01), "entropy_coefficient")
    config["advantage_clip_abs"] = _nonnegative_float(config.get("advantage_clip_abs", 0.0), "advantage_clip_abs")
    config["normalize_minibatch_advantages"] = bool(config.get("normalize_minibatch_advantages", False))
    config["loss_scale"] = _positive_float(config.get("loss_scale", 1.0), "loss_scale")
    config["max_grad_norm"] = _nonnegative_float(config.get("max_grad_norm", 1.0), "max_grad_norm")
    config["sampling_temperature"] = _positive_float(config.get("sampling_temperature", 1.0), "sampling_temperature")
    config["max_abs_approx_kl"] = _positive_float(config.get("max_abs_approx_kl", 0.5), "max_abs_approx_kl")
    config["canary_traffic_fraction"] = _nonnegative_float(config.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    config["require_d_drive_output_root"] = bool(config.get("require_d_drive_output_root", True))
    return config


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a positive integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a positive integer") from exc
    if numeric <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return numeric


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a non-negative integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a non-negative integer") from exc
    if numeric < 0:
        raise ConfigError(f"{name} must be a non-negative integer")
    return numeric


def _positive_float(value: Any, name: str) -> float:
    numeric = _finite(value)
    if numeric is None or numeric <= 0.0:
        raise ConfigError(f"{name} must be a positive finite number")
    return float(numeric)


def _nonnegative_float(value: Any, name: str) -> float:
    numeric = _finite(value)
    if numeric is None or numeric < 0.0:
        raise ConfigError(f"{name} must be a non-negative finite number")
    return float(numeric)


def _required_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name}_invalid")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}_invalid") from exc


def _required_float(value: Any, name: str) -> float:
    numeric = _finite(value)
    if numeric is None:
        raise ValueError(f"{name}_non_finite")
    return float(numeric)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _loss_row_finite(row: dict[str, Any]) -> bool:
    for key in ("total_loss", "policy_loss", "value_loss", "entropy", "pre_update_approx_kl", "post_update_approx_kl"):
        if _finite(row.get(key)) is None:
            return False
    return True


def _gradient_audit(model: XunceFullNetworkV1) -> dict[str, Any]:
    total_sq = 0.0
    parameter_with_grad_count = 0
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        parameter_with_grad_count += 1
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            finite = False
        total_sq += float(torch.sum(grad * grad))
    norm = math.sqrt(total_sq)
    return {
        "schema_version": "xunce-stage21-4-gradient-audit/v1",
        "parameter_with_grad_count": parameter_with_grad_count,
        "grad_norm": norm,
        "grad_norm_finite": bool(math.isfinite(norm) and finite),
        "pre_clip_grad_norm": None,
        "post_clip_grad_norm": None,
        "max_grad_norm": None,
    }


def _current_grad_norm(model: XunceFullNetworkV1) -> float:
    total_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total_sq += float(torch.sum(grad * grad))
    return math.sqrt(total_sq)


def _gradient_audit_passed(audit: dict[str, Any]) -> bool:
    return bool(audit.get("grad_norm_finite")) and int(audit.get("parameter_with_grad_count", 0)) > 0


def _parameter_delta_l2(model: XunceFullNetworkV1, initial_state: dict[str, torch.Tensor]) -> float:
    total_sq = 0.0
    current_state = model.state_dict()
    for key, initial in initial_state.items():
        delta = current_state[key].detach() - initial
        total_sq += float(torch.sum(delta * delta))
    return math.sqrt(total_sq)


def _sha256_file(path: Path) -> str | None:
    if not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty_checkpoint_audit() -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage21-4-checkpoint-audit/v1",
        "experimental_checkpoint_exists": False,
        "checkpoint_reload_passed": False,
    }


def _empty_gradient_audit() -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage21-4-gradient-audit/v1",
        "parameter_with_grad_count": 0,
        "grad_norm": None,
        "grad_norm_finite": False,
        "pre_clip_grad_norm": None,
        "post_clip_grad_norm": None,
        "max_grad_norm": None,
    }


def _unique_sorted(values: list[str]) -> list[str]:
    return sorted({str(value) for value in values if value})


if __name__ == "__main__":
    raise SystemExit(main())
