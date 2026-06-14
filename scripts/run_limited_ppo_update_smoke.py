from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from git_provenance import git_snapshot as _git_snapshot
from training_progress import add_progress_argument, make_progress_reporter, ppo_update_progress_metrics


CONFIG_SCHEMA_VERSION = "limited-ppo-update-smoke-config/v1"
SUMMARY_SCHEMA_VERSION = "limited-ppo-update-smoke-summary/v1"
CANDIDATE_SUMMARY_SCHEMA_VERSION = "raw-policy-generalization-candidate-summary/v1"
CHECKPOINT_SCHEMA_VERSION = "controlled-hybrid-policy-candidate-checkpoint/v1"
CHECKPOINT_METADATA_SCHEMA_VERSION = "controlled-hybrid-policy-candidate-checkpoint-metadata/v1"

NEXT_INPUT_INVALID = "limited_ppo_update_input_contract_invalid"
NEXT_NOT_ON_POLICY = "ppo_update_not_on_collector_policy"
NEXT_NON_FINITE = "ppo_update_loss_non_finite"
NEXT_UPDATE_TOO_LARGE = "ppo_update_too_large"
NEXT_CUDA_UNAVAILABLE = "cuda_requested_but_unavailable"
NEXT_DEVICE_CONTRACT_INVALID = "training_device_contract_invalid"
NEXT_DEVICE_TENSOR_MISMATCH = "device_tensor_mismatch"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a limited PPO update smoke test.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--base-candidate-root", required=True)
    parser.add_argument("--collector-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    add_progress_argument(parser)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": _display_path(config_path, repo_root)}, ensure_ascii=False))
        return 0

    summary = run_limited_ppo_update_smoke(
        source_root=_resolve_path(args.source_root, repo_root),
        base_candidate_root=_resolve_path(args.base_candidate_root, repo_root),
        collector_root=_resolve_path(args.collector_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
        progress_mode=args.progress,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "input_ppo_trainable_transition_count": summary["input_ppo_trainable_transition_count"],
                "optimizer_train_transition_count": summary["optimizer_train_transition_count"],
                "approx_kl": summary["approx_kl"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def run_limited_ppo_update_smoke(
    *,
    source_root: Path,
    base_candidate_root: Path,
    collector_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    progress_mode: str | None = None,
) -> dict[str, Any]:
    import torch
    from model_explorer.policy.architectures import build_policy_network_from_metadata
    from model_explorer.policy.dataset import validate_rollout_dataset
    from model_explorer.policy.device import resolve_training_device
    from model_explorer.policy.ppo import compute_masked_ppo_loss
    from model_explorer.policy.rollout import RolloutTransition
    from model_explorer.policy.rollout_io import read_rollout_episodes
    from model_explorer.policy.torch_policy import observation_to_tensors

    paths = _paths(
        source_root=source_root,
        base_candidate_root=base_candidate_root,
        collector_root=collector_root,
        output_root=output_root,
        config=config,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    progress = make_progress_reporter(output_root=output_root, mode=progress_mode, config=config)
    progress.emit(
        stage="ppo_update",
        status="start",
        current=0,
        total=_int_value(config["training"].get("epochs"), 1),
        message="limited PPO update smoke",
    )
    reason_codes: list[str] = []
    diagnostics: dict[str, Any] = {}
    device_resolution = resolve_training_device(config["training"].get("device"), torch_module=torch)
    device_summary = device_resolution.to_summary()
    for reason in device_resolution.reason_codes:
        _append_reason(reason_codes, reason)
    resolved_device = device_resolution.resolved_device
    collector_summary = _load_json(paths["collector_summary"], reason_codes=reason_codes, label="collector_summary")
    base_candidate_summary = _load_json(
        paths["base_candidate_summary"],
        reason_codes=reason_codes,
        label="base_candidate_summary",
    )
    base_metadata = _load_json(paths["base_checkpoint_metadata"], reason_codes=reason_codes, label="base_checkpoint_metadata")
    episodes = ()
    try:
        episodes = read_rollout_episodes(paths["rollout_episodes"])
        dataset_summary = validate_rollout_dataset(episodes)
    except Exception as exc:  # noqa: BLE001 - summary carries contract failures.
        dataset_summary = None
        diagnostics["dataset_error"] = str(exc)
        _append_reason(reason_codes, NEXT_INPUT_INVALID)

    all_transitions = tuple(transition for episode in episodes for transition in episode.transitions)
    trainable_transitions = tuple(
        transition
        for transition in all_transitions
        if _is_policy_ppo_trainable_transition(transition, config=config)
    )
    optimizer_transition_split_counts = dict(
        sorted(
            Counter(
                str(_transition_extra(transition).get("split") or "unknown")
                for transition in trainable_transitions
            ).items()
        )
    )
    optimizer_transition_source_counts = dict(
        sorted(
            Counter(
                str(_transition_extra(transition).get("controlled_choice_source") or "unknown")
                for transition in trainable_transitions
            ).items()
        )
    )
    allowed_sources = _trainable_filter_sources(config)
    validation_test_optimizer_transition_count = sum(
        1
        for transition in trainable_transitions
        if str(_transition_extra(transition).get("split") or "") in {"validation", "test"}
    )
    non_empty_gate_reason_optimizer_transition_count = sum(
        1
        for transition in trainable_transitions
        if _string_list(_transition_extra(transition).get("gate_reason_codes"))
    )
    disallowed_source_optimizer_transition_count = sum(
        1
        for transition in trainable_transitions
        if str(_transition_extra(transition).get("controlled_choice_source") or "") not in allowed_sources
    )
    source_fallback_trainable_count = sum(
        1
        for transition in all_transitions
        if transition.info.extra.get("controlled_choice_source") == "source_fallback"
        and transition.info.extra.get("ppo_trainable") is True
    )
    missing_context_id_count = sum(1 for transition in trainable_transitions if not transition.info.extra.get("context_id"))
    non_finite_reward_count = sum(1 for transition in trainable_transitions if not math.isfinite(float(transition.reward)))

    checkpoint: dict[str, Any] = {}
    network = None
    base_state = None
    old_log_prob_max_abs_error = math.inf
    old_value_max_abs_error = math.inf
    loss_non_finite_count = 0
    non_finite_gradient_count = 0
    non_finite_return_count = 0
    non_finite_advantage_count = 0
    parameter_l2_delta = 0.0
    approx_kl = math.inf
    clip_fraction = math.inf
    grad_norm_before_clip = math.inf
    max_grad_norm_after_clip = math.inf
    checkpoint_cpu_loadable = False
    training_curves: list[dict[str, Any]] = []
    training_result: dict[str, Any] | None = None

    if collector_summary.get("status") != "passed":
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if base_candidate_summary.get("status") != "passed":
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if base_metadata.get("schema_version") != CHECKPOINT_METADATA_SCHEMA_VERSION:
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    expected_count = _int_value(config["validation"].get("expected_input_ppo_trainable_transition_count"))
    if expected_count and _int_value(collector_summary.get("ppo_trainable_transition_count")) != expected_count:
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if source_fallback_trainable_count:
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if missing_context_id_count or non_finite_reward_count:
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if len(trainable_transitions) < _int_value(config["validation"].get("min_optimizer_train_transition_count")):
        _append_reason(reason_codes, NEXT_INPUT_INVALID)

    if trainable_transitions and not reason_codes:
        try:
            checkpoint = torch.load(paths["base_checkpoint"], map_location="cpu", weights_only=False)
            first = trainable_transitions[0].observation
            architecture = checkpoint.get("architecture") or base_metadata.get("architecture")
            hidden_size = _checkpoint_hidden_size(checkpoint, base_metadata, config)
            network = build_policy_network_from_metadata(
                architecture,
                candidate_feature_count=len(first.candidate_feature_names),
                global_feature_count=len(first.global_feature_names),
                missing_indicator_count=len(first.candidate_missing_indicator_names),
                hidden_size=hidden_size,
                architecture_config=checkpoint.get("architecture_config") or base_metadata.get("architecture_config"),
            )
            state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
            if not state:
                raise ValueError("base checkpoint is missing model state")
            network.load_state_dict(state)
            network.to(resolved_device)
            base_state = {key: value.detach().clone() for key, value in network.state_dict().items()}
            old_log_prob_max_abs_error, old_value_max_abs_error = _old_policy_errors(
                network,
                trainable_transitions,
                torch=torch,
                observation_to_tensors=observation_to_tensors,
                device=resolved_device,
            )
        except Exception as exc:  # noqa: BLE001
            diagnostics["checkpoint_error"] = str(exc)
            _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if trainable_transitions and not reason_codes:
        if old_log_prob_max_abs_error > float(config["validation"].get("max_old_log_prob_abs_error", 1.0e-4)):
            _append_reason(reason_codes, NEXT_NOT_ON_POLICY)
        if old_value_max_abs_error > float(config["validation"].get("max_old_value_abs_error", 1.0e-4)):
            _append_reason(reason_codes, NEXT_NOT_ON_POLICY)

    if network is not None and base_state is not None and not reason_codes:
        try:
            torch.manual_seed(_int_value(config["training"].get("seed")))
            batch = _ppo_batch(trainable_transitions, config=config, torch=torch, device=resolved_device)
            if _batch_device_mismatch(batch, resolved_device=resolved_device):
                _append_reason(reason_codes, NEXT_DEVICE_TENSOR_MISMATCH)
                raise ValueError("PPO batch tensors are not on the resolved training device")
            non_finite_return_count = _non_finite_tensor_count(batch["returns"], torch=torch)
            non_finite_advantage_count = _non_finite_tensor_count(batch["advantages"], torch=torch)
            if non_finite_return_count or non_finite_advantage_count:
                _append_reason(reason_codes, NEXT_NON_FINITE)
            optimizer = torch.optim.Adam(network.parameters(), lr=float(config["training"].get("learning_rate", 1.0e-5)))
            losses = None
            for epoch_index in range(_int_value(config["training"].get("epochs"), 1)):
                optimizer.zero_grad()
                losses = compute_masked_ppo_loss(
                    network,
                    candidate_features=batch["candidate_features"],
                    global_features=batch["global_features"],
                    action_mask=batch["action_mask"],
                    actions=batch["actions"],
                    old_log_probs=batch["old_log_probs"],
                    returns=batch["returns"],
                    advantages=batch["advantages"],
                    candidate_missing_indicators=batch["candidate_missing_indicators"],
                    clip_ratio=float(config["training"].get("clip_ratio", 0.2)),
                )
                if not _tensor_finite(losses.total_loss, torch=torch):
                    loss_non_finite_count += 1
                    _append_reason(reason_codes, NEXT_NON_FINITE)
                    break
                losses.total_loss.backward()
                grad_norm_before_clip, non_finite_gradient_count = _grad_norm(network, torch=torch)
                clip_value = float(config["training"].get("max_grad_norm", 1.0))
                torch.nn.utils.clip_grad_norm_(network.parameters(), clip_value)
                max_grad_norm_after_clip, post_clip_non_finite = _grad_norm(network, torch=torch)
                if max_grad_norm_after_clip <= clip_value + 1.0e-6:
                    max_grad_norm_after_clip = min(max_grad_norm_after_clip, clip_value)
                non_finite_gradient_count += post_clip_non_finite
                if non_finite_gradient_count:
                    _append_reason(reason_codes, NEXT_NON_FINITE)
                    break
                optimizer.step()
                approx_kl, clip_fraction = _ppo_diagnostics(
                    network,
                    batch,
                    torch=torch,
                    clip_ratio=float(config["training"].get("clip_ratio", 0.2)),
                )
                training_curves.append(
                    {
                        "epoch": epoch_index + 1,
                        "total_loss": _tensor_float(losses.total_loss),
                        "policy_loss": _tensor_float(losses.policy_loss),
                        "value_loss": _tensor_float(losses.value_loss),
                        "entropy": _tensor_float(losses.entropy),
                        "approx_kl": approx_kl,
                        "clip_fraction": clip_fraction,
                        "grad_norm_before_clip": grad_norm_before_clip,
                        "max_grad_norm_after_clip": max_grad_norm_after_clip,
                    }
                )
                progress.emit(
                    stage="ppo_update",
                    status="progress",
                    current=epoch_index + 1,
                    total=_int_value(config["training"].get("epochs"), 1),
                    message=f"ppo epoch {epoch_index + 1}/{_int_value(config['training'].get('epochs'), 1)}",
                    metrics={
                        "optimizer_train_transition_count": len(trainable_transitions),
                        "loss": training_curves[-1]["total_loss"],
                        "approx_kl": approx_kl,
                        "max_grad_norm_after_clip": max_grad_norm_after_clip,
                    },
                )
            parameter_l2_delta = _parameter_l2_delta(network.state_dict(), base_state, torch=torch)
            if parameter_l2_delta <= 0.0:
                _append_reason(reason_codes, NEXT_INPUT_INVALID)
            if approx_kl > float(config["validation"].get("max_approx_kl", 0.25)):
                _append_reason(reason_codes, NEXT_UPDATE_TOO_LARGE)
            if max_grad_norm_after_clip > float(config["validation"].get("max_grad_norm_after_clip", 1.0)) + 1.0e-6:
                _append_reason(reason_codes, NEXT_UPDATE_TOO_LARGE)
            if loss_non_finite_count or non_finite_gradient_count:
                _append_reason(reason_codes, NEXT_NON_FINITE)
            if not reason_codes:
                training_result = {
                    "sample_count": len(trainable_transitions),
                    "epochs": _int_value(config["training"].get("epochs"), 1),
                    "learning_rate": float(config["training"].get("learning_rate", 1.0e-5)),
                    "clip_ratio": float(config["training"].get("clip_ratio", 0.2)),
                    "discount_factor": float(config["training"].get("discount_factor", 0.99)),
                    "final_total_loss": training_curves[-1]["total_loss"],
                    "final_policy_loss": training_curves[-1]["policy_loss"],
                    "final_value_loss": training_curves[-1]["value_loss"],
                    "final_entropy": training_curves[-1]["entropy"],
                    "device": device_summary,
                }
                _write_checkpoint_outputs(
                    network=network,
                    checkpoint=checkpoint,
                    base_metadata=base_metadata,
                    config=config,
                    paths=paths,
                    output_root=output_root,
                    source_root=source_root,
                    base_candidate_root=base_candidate_root,
                    collector_root=collector_root,
                    repo_root=repo_root,
                    sample_count=len(trainable_transitions),
                    training_result=training_result,
                    device_summary=device_summary,
                )
                checkpoint_cpu_loadable = _checkpoint_cpu_loadable(paths["checkpoint"], torch=torch)
                if not checkpoint_cpu_loadable:
                    _append_reason(reason_codes, NEXT_INPUT_INVALID)
        except Exception as exc:  # noqa: BLE001
            diagnostics["training_error"] = str(exc)
            _append_reason(reason_codes, NEXT_NON_FINITE)

    status = "failed" if reason_codes else "passed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "next_required_change": _next_required_change(reason_codes),
        "source_root": _display_path(source_root, repo_root),
        "base_candidate_root": _display_path(base_candidate_root, repo_root),
        "collector_root": _display_path(collector_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "requested_device": device_summary["requested_device"],
        "resolved_device": device_summary["resolved_device"],
        "cuda_available": device_summary["cuda_available"],
        "cuda_device_name": device_summary["cuda_device_name"],
        "fallback_to_cpu": device_summary["fallback_to_cpu"],
        "device_reason_codes": list(device_resolution.reason_codes),
        "input_ppo_trainable_transition_count": len(trainable_transitions),
        "optimizer_train_transition_count": len(trainable_transitions) if training_result else 0,
        "optimizer_transition_split_counts": optimizer_transition_split_counts,
        "optimizer_transition_source_counts": optimizer_transition_source_counts,
        "validation_test_optimizer_transition_count": validation_test_optimizer_transition_count,
        "non_empty_gate_reason_optimizer_transition_count": non_empty_gate_reason_optimizer_transition_count,
        "disallowed_source_optimizer_transition_count": disallowed_source_optimizer_transition_count,
        "collector_ppo_trainable_transition_count": _int_value(collector_summary.get("ppo_trainable_transition_count")),
        "source_fallback_trainable_count": source_fallback_trainable_count,
        "missing_context_id_count": missing_context_id_count,
        "old_log_prob_max_abs_error": _finite_or_inf(old_log_prob_max_abs_error),
        "old_value_max_abs_error": _finite_or_inf(old_value_max_abs_error),
        "loss_non_finite_count": loss_non_finite_count,
        "non_finite_gradient_count": non_finite_gradient_count,
        "non_finite_reward_count": non_finite_reward_count,
        "non_finite_return_count": non_finite_return_count,
        "non_finite_advantage_count": non_finite_advantage_count,
        "parameter_l2_delta": parameter_l2_delta,
        "approx_kl": _finite_or_inf(approx_kl),
        "clip_fraction": _finite_or_inf(clip_fraction),
        "grad_norm_before_clip": _finite_or_inf(grad_norm_before_clip),
        "max_grad_norm_after_clip": _finite_or_inf(max_grad_norm_after_clip),
        "checkpoint_cpu_loadable": checkpoint_cpu_loadable,
        "training_result": training_result,
        "dataset_summary": dataset_summary,
        "summary": _display_path(paths["summary"], repo_root),
        "training_curves": _display_path(paths["training_curves"], repo_root),
        "diagnostics": _display_path(paths["diagnostics"], repo_root),
        "checkpoint_path": _display_path(paths["checkpoint"], repo_root),
        "checkpoint_metadata_path": _display_path(paths["checkpoint_metadata"], repo_root),
        "candidate_summary_path": _display_path(paths["candidate_summary"], repo_root),
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "runs_formal_ppo_rollout": False,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }
    _write_json(paths["summary"], summary)
    _write_json(paths["training_curves"], {"schema_version": "limited-ppo-update-training-curves/v1", "records": training_curves})
    _write_json(paths["diagnostics"], {"schema_version": "limited-ppo-update-diagnostics/v1", **diagnostics})
    progress.emit(
        stage="ppo_update",
        status=status,
        current=_int_value(config["training"].get("epochs"), 1),
        total=_int_value(config["training"].get("epochs"), 1),
        message=f"limited PPO update {status}",
        summary_path=_display_path(paths["summary"], repo_root),
        reason_codes=reason_codes,
        metrics=ppo_update_progress_metrics(summary),
    )
    progress.finalize(status=status)
    return summary


def _is_policy_ppo_trainable_transition(transition, *, config: dict[str, Any] | None = None) -> bool:
    extra = _transition_extra(transition)
    allowed_sources = _trainable_filter_sources(config)
    allowed_splits = _trainable_filter_splits(config)
    return (
        transition.action_index >= 0
        and extra.get("ppo_trainable") is True
        and str(extra.get("controlled_choice_source") or "") in allowed_sources
        and (allowed_splits is None or str(extra.get("split") or "") in allowed_splits)
        and (
            not _trainable_filter_requires_empty_gate_reasons(config)
            or not _string_list(extra.get("gate_reason_codes"))
        )
    )


def _transition_extra(transition) -> dict[str, Any]:
    info = getattr(transition, "info", None)
    extra = getattr(info, "extra", None)
    return extra if isinstance(extra, dict) else {}


def _trainable_filter_sources(config: dict[str, Any] | None) -> set[str]:
    filter_config = config.get("trainable_filter", {}) if isinstance(config, dict) else {}
    sources = filter_config.get("controlled_choice_sources", ["policy"])
    if not isinstance(sources, list) or not sources:
        return {"policy"}
    return {str(source) for source in sources}


def _trainable_filter_splits(config: dict[str, Any] | None) -> set[str] | None:
    filter_config = config.get("trainable_filter", {}) if isinstance(config, dict) else {}
    splits = filter_config.get("splits")
    if splits is None:
        return None
    if not isinstance(splits, list) or not splits:
        return set()
    return {str(split) for split in splits}


def _trainable_filter_requires_empty_gate_reasons(config: dict[str, Any] | None) -> bool:
    filter_config = config.get("trainable_filter", {}) if isinstance(config, dict) else {}
    return bool(filter_config.get("require_empty_gate_reason_codes", False))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _old_policy_errors(network, transitions, *, torch, observation_to_tensors, device: str) -> tuple[float, float]:
    log_errors: list[float] = []
    value_errors: list[float] = []
    network.eval()
    with torch.no_grad():
        for transition in transitions:
            tensors = observation_to_tensors(transition.observation, device=device)
            output = network(**tensors)
            distribution = torch.distributions.Categorical(logits=output.masked_logits)
            action = torch.tensor([transition.action_index], dtype=torch.long, device=device)
            log_prob = float(distribution.log_prob(action).item())
            value = float(output.value[0].item())
            log_errors.append(abs(log_prob - float(transition.log_prob)))
            value_errors.append(abs(value - float(transition.value)))
    return max(log_errors) if log_errors else math.inf, max(value_errors) if value_errors else math.inf


def _ppo_batch(transitions, *, config: dict[str, Any], torch, device: str | None = None) -> dict[str, Any]:
    action_count = max(len(transition.observation.action_mask) for transition in transitions)
    rewards = [float(transition.reward) for transition in transitions]
    dones = [bool(transition.done) for transition in transitions]
    old_values = [float(transition.value) for transition in transitions]
    returns, advantages = _ppo_returns_and_advantages(
        transitions,
        rewards=rewards,
        dones=dones,
        old_values=old_values,
        config=config,
    )
    return {
        "candidate_features": torch.tensor(
            [_padded_candidate_features(transition.observation, action_count) for transition in transitions],
            dtype=torch.float32,
            device=device,
        ),
        "global_features": torch.tensor(
            [transition.observation.global_features for transition in transitions],
            dtype=torch.float32,
            device=device,
        ),
        "action_mask": torch.tensor(
            [_padded_action_mask(transition.observation, action_count) for transition in transitions],
            dtype=torch.bool,
            device=device,
        ),
        "candidate_missing_indicators": torch.tensor(
            [_padded_missing_indicators(transition.observation, action_count) for transition in transitions],
            dtype=torch.float32,
            device=device,
        ),
        "actions": torch.tensor([transition.action_index for transition in transitions], dtype=torch.long, device=device),
        "old_log_probs": torch.tensor(
            [float(transition.log_prob) for transition in transitions],
            dtype=torch.float32,
            device=device,
        ),
        "returns": torch.tensor(returns, dtype=torch.float32, device=device),
        "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
    }


def _ppo_returns_and_advantages(
    transitions,
    *,
    rewards: list[float],
    dones: list[bool],
    old_values: list[float],
    config: dict[str, Any],
) -> tuple[list[float], list[float]]:
    training = config.get("training", {}) if isinstance(config, dict) else {}
    if training.get("return_source") == "transition_info":
        return_field = str(training.get("return_field") or "ppo_return")
        advantage_field = str(training.get("advantage_field") or "ppo_advantage")
        returns: list[float] = []
        advantages: list[float] = []
        for transition in transitions:
            extra = _transition_extra(transition)
            returns.append(_float_or_nan(extra.get(return_field)))
            advantages.append(_float_or_nan(extra.get(advantage_field)))
        return returns, advantages

    returns = _discounted_returns(
        rewards,
        dones=dones,
        discount_factor=float(training.get("discount_factor", 0.99)),
    )
    advantages = [return_value - value for return_value, value in zip(returns, old_values)]
    return returns, advantages


def _discounted_returns(rewards: list[float], *, dones: list[bool], discount_factor: float) -> list[float]:
    values = [0.0 for _ in rewards]
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if dones[index]:
            running = 0.0
        running = rewards[index] + discount_factor * running
        values[index] = running
    return values


def _padded_candidate_features(observation, action_count: int) -> list[list[float]]:
    width = len(observation.candidate_feature_names)
    rows = [list(row) for row in observation.candidate_features]
    rows.extend([[0.0 for _ in range(width)] for _ in range(action_count - len(rows))])
    return rows


def _padded_action_mask(observation, action_count: int) -> list[bool]:
    values = list(observation.action_mask)
    values.extend(False for _ in range(action_count - len(values)))
    return values


def _padded_missing_indicators(observation, action_count: int) -> list[list[float]]:
    width = len(observation.candidate_missing_indicator_names)
    rows = [list(row) for row in observation.candidate_missing_indicators]
    if not rows:
        rows = [[0.0 for _ in range(width)] for _ in observation.action_mask]
    rows.extend([[0.0 for _ in range(width)] for _ in range(action_count - len(rows))])
    return rows


def _ppo_diagnostics(network, batch: dict[str, Any], *, torch, clip_ratio: float) -> tuple[float, float]:
    with torch.no_grad():
        output = network(
            candidate_features=batch["candidate_features"],
            global_features=batch["global_features"],
            action_mask=batch["action_mask"],
            candidate_missing_indicators=batch["candidate_missing_indicators"],
        )
        distribution = torch.distributions.Categorical(logits=output.masked_logits)
        new_log_probs = distribution.log_prob(batch["actions"])
        diff = new_log_probs - batch["old_log_probs"]
        ratios = torch.exp(diff)
        approx_kl = torch.mean(batch["old_log_probs"] - new_log_probs)
        clip_fraction = torch.mean((torch.abs(ratios - 1.0) > clip_ratio).to(torch.float32))
    return _tensor_float(approx_kl), _tensor_float(clip_fraction)


def _grad_norm(network, *, torch) -> tuple[float, int]:
    total_sq = 0.0
    non_finite = 0
    for parameter in network.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            non_finite += 1
            continue
        total_sq += float(torch.sum(grad * grad).item())
    return total_sq ** 0.5, non_finite


def _parameter_l2_delta(state: dict[str, Any], base_state: dict[str, Any], *, torch) -> float:
    total_sq = 0.0
    for key, value in state.items():
        base = base_state[key]
        delta = value.detach() - base.to(device=value.device)
        total_sq += float(torch.sum(delta * delta).item())
    return total_sq ** 0.5


def _cpu_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu() if hasattr(value, "detach") else value
        for key, value in state_dict.items()
    }


def _batch_device_mismatch(batch: dict[str, Any], *, resolved_device: str) -> bool:
    for value in batch.values():
        if hasattr(value, "device") and value.device.type != resolved_device:
            return True
    return False


def _checkpoint_cpu_loadable(path: Path, *, torch) -> bool:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - caller records contract failure.
        return False
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if not isinstance(state, dict):
        return False
    return all(not hasattr(value, "device") or value.device.type == "cpu" for value in state.values())


def _write_checkpoint_outputs(
    *,
    network,
    checkpoint: dict[str, Any],
    base_metadata: dict[str, Any],
    config: dict[str, Any],
    paths: dict[str, Path],
    output_root: Path,
    source_root: Path,
    base_candidate_root: Path,
    collector_root: Path,
    repo_root: Path,
    sample_count: int,
    training_result: dict[str, Any],
    device_summary: dict[str, Any],
) -> None:
    import torch

    git_provenance = {"current": _git_snapshot(repo_root), "current_matches_sources": True}
    training = dict(config["training"])
    training["device"] = device_summary.get("requested_device", training.get("device", "cpu"))
    training["device_resolution"] = device_summary
    updated_checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experimental": True,
        "architecture": network.architecture_name,
        "model_state_dict": _cpu_state_dict(network.state_dict()),
        "training": training,
        "device": device_summary,
        "source_root": _display_path(source_root, repo_root),
        "base_candidate_root": _display_path(base_candidate_root, repo_root),
        "collector_root": _display_path(collector_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "git_provenance": git_provenance,
        "device": device_summary,
    }
    torch.save(updated_checkpoint, paths["checkpoint"])
    metadata = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "experimental": True,
        "checkpoint_path": _display_path(paths["checkpoint"], repo_root),
        "architecture": network.architecture_name,
        "sample_count": int(sample_count),
        "ppo_update_sample_count": int(sample_count),
        "epochs": _int_value(training.get("epochs"), 1),
        "seed": _int_value(training.get("seed")),
        "hidden_size": _checkpoint_hidden_size(checkpoint, base_metadata, config),
        "learning_rate": float(training.get("learning_rate", 1.0e-5)),
        "clip_ratio": float(training.get("clip_ratio", 0.2)),
        "device": device_summary,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "git_provenance": git_provenance,
    }
    _write_json(paths["checkpoint_metadata"], metadata)
    candidate_summary = {
        "schema_version": CANDIDATE_SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": "passed",
        "reason_codes": [],
        "checkpoint_path": _display_path(paths["checkpoint"], repo_root),
        "checkpoint_metadata_path": _display_path(paths["checkpoint_metadata"], repo_root),
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "candidate_training_status": "passed",
        "training_result": training_result,
        "train_pair_count": int(sample_count),
        "leaked_context_id_count": 0,
        "git_provenance": git_provenance,
        "device": device_summary,
        "non_goals": list(config.get("non_goals", [])),
    }
    _write_json(paths["candidate_summary"], candidate_summary)


def _paths(
    *,
    source_root: Path,
    base_candidate_root: Path,
    collector_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Path]:
    inputs = config["input_files"]
    outputs = config["output_files"]
    return {
        "source_root": source_root,
        "rollout_episodes": collector_root / inputs["rollout_episodes"],
        "collector_summary": collector_root / inputs["collector_summary"],
        "base_checkpoint": base_candidate_root / inputs["base_checkpoint"],
        "base_checkpoint_metadata": base_candidate_root / inputs["base_checkpoint_metadata"],
        "base_candidate_summary": base_candidate_root / inputs["base_candidate_summary"],
        "summary": output_root / outputs["summary"],
        "training_curves": output_root / outputs["training_curves"],
        "diagnostics": output_root / outputs["diagnostics"],
        "checkpoint": output_root / outputs["checkpoint"],
        "checkpoint_metadata": output_root / outputs["checkpoint_metadata"],
        "candidate_summary": output_root / outputs["candidate_summary"],
    }


def _checkpoint_hidden_size(checkpoint: dict[str, Any], metadata: dict[str, Any], config: dict[str, Any]) -> int:
    training = checkpoint.get("training") if isinstance(checkpoint.get("training"), dict) else {}
    return _int_value(
        training.get("hidden_size")
        or checkpoint.get("hidden_size")
        or metadata.get("hidden_size")
        or config.get("evaluation", {}).get("hidden_size")
        or config["training"].get("hidden_size")
        or 16,
        16,
    )


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
    for section in ("input_files", "output_files", "training", "validation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _load_json(path: Path, *, reason_codes: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
        return {}
    if not isinstance(payload, dict):
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
        return {}
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _next_required_change(reason_codes: list[str]) -> str | None:
    if not reason_codes:
        return None
    for reason in (
        NEXT_CUDA_UNAVAILABLE,
        NEXT_DEVICE_CONTRACT_INVALID,
        NEXT_DEVICE_TENSOR_MISMATCH,
        NEXT_INPUT_INVALID,
        NEXT_NOT_ON_POLICY,
        NEXT_NON_FINITE,
        NEXT_UPDATE_TOO_LARGE,
    ):
        if reason in reason_codes:
            return reason
    return reason_codes[0]


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite_or_inf(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("inf")


def _tensor_float(value) -> float:
    return float(value.detach().cpu())


def _non_finite_tensor_count(value, *, torch) -> int:
    return int((~torch.isfinite(value)).sum().item())


def _tensor_finite(value, *, torch) -> bool:
    return bool(torch.isfinite(value).all().item())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
