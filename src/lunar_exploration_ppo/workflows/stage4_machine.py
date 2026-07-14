"""Stage 4 real Smoke PPO and tiny-overfit machine acceptance."""

from __future__ import annotations

import io
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

import numpy as np
import torch

from lunar_exploration_ppo.configs.stage1 import Stage1Config
from lunar_exploration_ppo.configs.stage4 import Stage4Config
from lunar_exploration_ppo.policy.cross_attention import (
    CrossAttentionFrontierPolicy,
    batch_policy_observations,
    sample_action,
)
from lunar_exploration_ppo.policy.observation import PolicyObservation
from lunar_exploration_ppo.ppo.checkpoint import CheckpointManager
from lunar_exploration_ppo.ppo.collector import (
    CollectorContract,
    RolloutCollector,
    SpawnVectorEnv,
    lunar_env_specs,
)
from lunar_exploration_ppo.ppo.rollout import RolloutContractError
from lunar_exploration_ppo.ppo.resume import (
    optimizer_state_sha256,
    resume_training_from_last_checkpoint,
)
from lunar_exploration_ppo.ppo.tiny_overfit import run_tiny_overfit_acceptance
from lunar_exploration_ppo.ppo.trainer import PPOTrainer


STAGE4_SMOKE_UPDATE_SCHEMA_VERSION: Final = "stage4_smoke_update_audit/v1"
STAGE4_ACTION_FIXTURE_SCHEMA_VERSION: Final = (
    "stage4_checkpoint_action_fixture/v1"
)


class Stage4MachineError(RuntimeError):
    """The real Stage 4 machine acceptance failed closed."""


@dataclass(frozen=True, slots=True)
class Stage4MachineAuditBundle:
    updates: tuple[dict[str, object], ...]
    tiny_overfit: dict[str, object]
    checkpoint_action_fixture_npz: bytes
    training_resume: dict[str, object]


class _MachineJsonStateComponent:
    """Stage 4 disabled normalizer / aggregate sampler 的实际 runtime state。"""

    def __init__(self, state: Mapping[str, object]) -> None:
        self._state = _copy_json_mapping(state)

    def restore_state(self, state: dict[str, object]) -> None:
        self._state = _copy_json_mapping(state)

    def capture_state(self) -> dict[str, object]:
        return _copy_json_mapping(self._state)


def run_stage4_smoke_acceptance(
    *,
    config: Stage4Config,
    stage1_config: Stage1Config,
    checkpoint_root: str | Path,
    config_sha256: str,
    lineage: Mapping[str, object],
) -> Stage4MachineAuditBundle:
    """Run three real 8x128 Smoke updates, checkpoint round trips, and tiny PPO."""

    if config.device != "cuda" or config.allow_cpu_fallback:
        raise Stage4MachineError("Stage 4 machine acceptance requires CUDA only")
    if not torch.cuda.is_available():
        raise Stage4MachineError("CUDA is unavailable; CPU fallback is forbidden")
    if (
        config.acceptance.smoke_updates != 3
        or config.acceptance.trainable_transitions_per_update != 1024
    ):
        raise Stage4MachineError("Stage 4 Smoke acceptance config drift")

    random.seed(config.training_seed)
    np.random.seed(config.training_seed)
    torch.manual_seed(config.training_seed)
    torch.cuda.manual_seed_all(config.training_seed)
    device = torch.device("cuda")
    policy = CrossAttentionFrontierPolicy().to(device=device, dtype=torch.float32)
    trainer = PPOTrainer(
        policy,
        device=device,
        shuffle_seed=config.training_seed,
    )
    checkpoint_manager = CheckpointManager(checkpoint_root)
    contract = CollectorContract(
        config_sha256=config_sha256,
        lineage=lineage,
        frontier_extractor_version=config.frontier_extractor_version,
        observation_schema_version=config.observation_schema_version,
        action_space_version=config.action_space_version,
        reward_version=config.reward_version,
        planner_version=config.planner_version,
    )
    versions = {
        "observation_schema_version": config.observation_schema_version,
        "action_space_version": config.action_space_version,
        "network_architecture_version": config.network_architecture_version,
        "reward_version": config.reward_version,
        "frontier_version": config.frontier_extractor_version,
        "planner_version": config.planner_version,
    }
    top_m_config = {
        "frontier_top_m": config.frontier_top_m,
        "selection": "stage2_observed_frontier_top_m/v1",
    }
    training_config = {
        "rollout": config.rollout.model_dump(mode="json"),
        "ppo": config.ppo.model_dump(mode="json"),
        "acceptance": config.acceptance.model_dump(mode="json"),
    }

    updates: list[dict[str, object]] = []
    final_fixture: PolicyObservation | None = None
    final_action: dict[str, object] | None = None
    training_resume: dict[str, object] | None = None
    specs = lunar_env_specs(stage1_config.model_dump(mode="json"))
    vector_env = SpawnVectorEnv(
        specs,
        timeout_seconds=config.worker_timeout_seconds,
    )
    continue_from_current_state = False
    try:
        for expected_step in range(1, config.acceptance.smoke_updates + 1):
            collector = RolloutCollector(
                policy=policy,
                vector_env=vector_env,
                device=device,
                contract=contract,
            )
            collection = (
                collector.collect(continue_from_current_state=True)
                if continue_from_current_state
                else collector.collect()
            )
            batch = collection.batch
            transitions = batch.transitions
            if len(transitions) != config.acceptance.trainable_transitions_per_update:
                raise Stage4MachineError("Smoke rollout transition count drift")
            action_fixture = transitions[0].snapshot.to_policy_observation()
            gae_audit = {
                "finite": bool(
                    np.isfinite(batch.raw_advantages).all()
                    and np.isfinite(batch.normalized_advantages).all()
                    and np.isfinite(batch.returns).all()
                ),
                "normalized_advantage_mean": float(
                    batch.normalized_advantages.mean(dtype=np.float64)
                ),
                "normalized_advantage_std": float(
                    batch.normalized_advantages.std(dtype=np.float64, ddof=0)
                ),
                "raw_advantage_min": float(batch.raw_advantages.min()),
                "raw_advantage_max": float(batch.raw_advantages.max()),
                "return_min": float(batch.returns.min()),
                "return_max": float(batch.returns.max()),
            }
            factorization_exact = all(
                transition.old_log_prob_total.tobytes()
                == np.float32(
                    transition.old_log_prob_frontier
                    + transition.old_log_prob_theta
                ).tobytes()
                for transition in transitions
            )
            ppo_metrics = trainer.update(batch)
            if ppo_metrics.update_step != expected_step:
                raise Stage4MachineError("PPO update step drift")
            cleared = False
            try:
                _ = batch.transitions
            except RolloutContractError:
                cleared = True
            if not batch.consumed or not cleared:
                raise Stage4MachineError("PPO rollout batch was not consumed and cleared")

            current_action = _deterministic_action_record(
                policy,
                action_fixture,
                device=device,
            )
            normalization_state = {
                "schema_version": "stage4_no_running_normalizer/v1",
                "enabled": False,
            }
            scenario_sampler_state = {
                "schema_version": "stage4_fixed_smoke_sampler_state/v1",
                "worker_sampler_states": [
                    state["sampler_state"]
                    for state in collection.vector_env_states
                ],
            }
            receipt = checkpoint_manager.save_complete(
                policy=policy,
                optimizer=trainer.optimizer,
                update_step=expected_step,
                normalization_stats=normalization_state,
                scenario_sampler_state=scenario_sampler_state,
                vector_env_states=collection.vector_env_states,
                best_record={
                    "schema_version": "stage4_no_performance_best_record/v1",
                    "update_step": expected_step,
                    "claim_boundary": "training_mechanics_only/v1",
                },
                versions=versions,
                top_m_config=top_m_config,
                scale_profile=config.scale_profile,
                training_config=training_config,
                config_sha256=config_sha256,
                lineage=lineage,
                eval_metrics={
                    "schema_version": "stage4_no_task_performance_eval/v1",
                    "claim_boundary": "ppo_training_system_closure_only/v1",
                },
            )
            loaded_policy = CrossAttentionFrontierPolicy().to(
                device=device,
                dtype=torch.float32,
            )
            loaded_optimizer = torch.optim.AdamW(
                loaded_policy.parameters(),
                lr=3.0e-4,
                eps=1.0e-5,
                weight_decay=1.0e-4,
            )
            loaded = checkpoint_manager.load_last_complete(
                policy=loaded_policy,
                optimizer=loaded_optimizer,
                expected_config_sha256=config_sha256,
                expected_lineage=lineage,
            )
            loaded_action = _deterministic_action_record(
                loaded_policy,
                action_fixture,
                device=device,
            )
            action_bit_exact = loaded_action == current_action
            if (
                loaded.update_step != expected_step
                or loaded.policy_state_sha256 != receipt.policy_state_sha256
                or not action_bit_exact
            ):
                raise Stage4MachineError("checkpoint deterministic load audit failed")

            if training_resume is not None and expected_step == 2:
                resume_receipt = training_resume.get("resume_receipt")
                if (
                    not isinstance(resume_receipt, dict)
                    or resume_receipt.get("next_update_step") != expected_step
                    or collection.audit.policy_state_sha256
                    != resume_receipt.get("policy_state_sha256")
                    or ppo_metrics.policy_state_sha256_before
                    != resume_receipt.get("policy_state_sha256")
                ):
                    raise Stage4MachineError(
                        "resumed update did not start from checkpoint policy"
                    )
                training_resume.update(
                    {
                        "continued_from_current_state": True,
                        "next_update_executed": True,
                        "next_update_step": ppo_metrics.update_step,
                        "collection_policy_state_sha256": (
                            collection.audit.policy_state_sha256
                        ),
                        "policy_state_sha256_before_next_update": (
                            ppo_metrics.policy_state_sha256_before
                        ),
                        "policy_state_sha256_after_next_update": (
                            ppo_metrics.policy_state_sha256_after
                        ),
                        "optimizer_state_sha256_after_next_update": (
                            optimizer_state_sha256(trainer.optimizer)
                        ),
                    }
                )

            update_record = {
                "schema_version": STAGE4_SMOKE_UPDATE_SCHEMA_VERSION,
                "update_step": expected_step,
                "config_sha256": config_sha256,
                "collection": {
                    "trainable_transition_count": (
                        collection.audit.trainable_transition_count
                    ),
                    "per_env_trainable_counts": list(
                        collection.audit.per_env_trainable_counts
                    ),
                    "diagnostic_reset_counts": list(
                        collection.audit.diagnostic_reset_counts
                    ),
                    "worker_pids": list(collection.audit.worker_pids),
                    "worker_start_methods": list(
                        collection.audit.worker_start_methods
                    ),
                    "inference_pids": list(collection.audit.inference_pids),
                    "inference_main_process_only": (
                        collection.audit.inference_pids == (os.getpid(),)
                    ),
                    "inference_batch_count": collection.audit.inference_batch_count,
                    "policy_device": collection.audit.policy_device,
                    "policy_state_sha256": collection.audit.policy_state_sha256,
                    "snapshot_count": len(collection.audit.snapshot_sha256),
                    "unique_snapshot_count": len(
                        set(collection.audit.snapshot_sha256)
                    ),
                    "terminal_transition_count": (
                        collection.audit.terminal_transition_count
                    ),
                    "old_logprob_factorization_exact": factorization_exact,
                },
                "gae": gae_audit,
                "ppo": _ppo_metrics_record(ppo_metrics),
                "buffer": {"consumed": batch.consumed, "cleared": cleared},
                "checkpoint": {
                    "update_step": receipt.update_step,
                    "checkpoint_sha256": receipt.checkpoint_sha256,
                    "manifest_sha256": receipt.manifest_sha256,
                    "loaded_last_complete_step": loaded.update_step,
                    "policy_state_sha256": receipt.policy_state_sha256,
                    "deterministic_action_bit_exact": action_bit_exact,
                    "deterministic_action": current_action,
                },
            }
            if not _finite_tree(update_record):
                raise Stage4MachineError("Smoke update audit contains non-finite data")
            updates.append(update_record)
            final_fixture = action_fixture
            final_action = current_action
            del (
                loaded_optimizer,
                loaded_policy,
                collection,
                transitions,
                batch,
                collector,
            )

            if expected_step == 1:
                vector_env.close()
                del trainer, policy
                torch.cuda.empty_cache()
                policy = CrossAttentionFrontierPolicy().to(
                    device=device,
                    dtype=torch.float32,
                )
                trainer = PPOTrainer(
                    policy,
                    device=device,
                    shuffle_seed=config.training_seed,
                )
                fresh_trainer_initial_update_step = trainer.update_step
                vector_env = SpawnVectorEnv(
                    specs,
                    timeout_seconds=config.worker_timeout_seconds,
                )
                normalizer = _MachineJsonStateComponent(
                    {"schema_version": "unrestored_normalizer/v1"}
                )
                scenario_sampler = _MachineJsonStateComponent(
                    {"schema_version": "unrestored_sampler/v1"}
                )
                resume_receipt = resume_training_from_last_checkpoint(
                    checkpoint_manager=checkpoint_manager,
                    trainer=trainer,
                    normalizer=normalizer,
                    scenario_sampler=scenario_sampler,
                    vector_env=vector_env,
                    expected_config_sha256=config_sha256,
                    expected_lineage=lineage,
                )
                if (
                    fresh_trainer_initial_update_step != 0
                    or resume_receipt.update_step != expected_step
                    or resume_receipt.checkpoint_sha256
                    != receipt.checkpoint_sha256
                    or resume_receipt.manifest_sha256 != receipt.manifest_sha256
                    or normalizer.capture_state() != normalization_state
                    or scenario_sampler.capture_state()
                    != scenario_sampler_state
                ):
                    raise Stage4MachineError(
                        "training runtime resume application failed"
                    )
                training_resume = {
                    "schema_version": "stage4_training_resume_audit/v1",
                    "fresh_trainer_initial_update_step": (
                        fresh_trainer_initial_update_step
                    ),
                    "resume_receipt": resume_receipt.to_dict(),
                    "normalization_state_applied": True,
                    "scenario_sampler_state_applied": True,
                    "vector_env_states_applied": True,
                    "vector_env_state_count": len(
                        vector_env.capture_states()
                    ),
                    "continued_from_current_state": False,
                    "next_update_executed": False,
                    "next_update_step": None,
                    "collection_policy_state_sha256": None,
                    "policy_state_sha256_before_next_update": None,
                    "policy_state_sha256_after_next_update": None,
                    "optimizer_state_sha256_after_next_update": None,
                }
                continue_from_current_state = True
    finally:
        vector_env.close()

    if training_resume is None or training_resume.get("next_update_executed") is not True:
        raise Stage4MachineError("training resume did not execute update N+1")

    tiny = run_tiny_overfit_acceptance(device=device).to_dict()
    if tiny.get("passed") is not True:
        raise Stage4MachineError("three-seed tiny-overfit acceptance failed")
    if final_fixture is None or final_action is None:
        raise Stage4MachineError("checkpoint action fixture was not captured")
    fixture_npz = _action_fixture_npz(final_fixture, final_action)
    return Stage4MachineAuditBundle(
        updates=tuple(updates),
        tiny_overfit=tiny,
        checkpoint_action_fixture_npz=fixture_npz,
        training_resume=training_resume,
    )


def load_action_fixture_npz(payload: bytes) -> tuple[PolicyObservation, dict[str, object]]:
    expected = {
        "schema_json",
        "expected_action_json",
        "prior_channels",
        "coverage_summary",
        "local_crop",
        "frontier_features",
        "pose_features",
        "candidate_mask",
    }
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise Stage4MachineError("checkpoint action fixture member drift")
            schema = json.loads(str(archive["schema_json"].item()))
            action = json.loads(str(archive["expected_action_json"].item()))
            observation = PolicyObservation(
                prior_channels=np.asarray(
                    archive["prior_channels"], dtype=np.float32
                ).copy(),
                coverage_summary=np.asarray(
                    archive["coverage_summary"], dtype=np.float32
                ).copy(),
                local_crop=np.asarray(archive["local_crop"], dtype=np.float32).copy(),
                frontier_features=np.asarray(
                    archive["frontier_features"], dtype=np.float32
                ).copy(),
                pose_features=np.asarray(
                    archive["pose_features"], dtype=np.float32
                ).copy(),
                candidate_mask=np.asarray(
                    archive["candidate_mask"], dtype=bool
                ).copy(),
            )
    except Stage4MachineError:
        raise
    except Exception as exc:
        raise Stage4MachineError("checkpoint action fixture cannot be decoded") from exc
    if schema != {
        "schema_version": STAGE4_ACTION_FIXTURE_SCHEMA_VERSION,
        "observation_schema_version": PolicyObservation.schema_version,
    } or not isinstance(action, dict):
        raise Stage4MachineError("checkpoint action fixture schema drift")
    return observation, action


def deterministic_action_record(
    policy: CrossAttentionFrontierPolicy,
    observation: PolicyObservation,
    *,
    device: torch.device | str,
) -> dict[str, object]:
    return _deterministic_action_record(policy, observation, device=torch.device(device))


def _deterministic_action_record(
    policy: CrossAttentionFrontierPolicy,
    observation: PolicyObservation,
    *,
    device: torch.device,
) -> dict[str, object]:
    policy.eval()
    with torch.no_grad():
        batch = batch_policy_observations((observation,), device=device)
        output = policy(batch)
        action = sample_action(output, batch.candidate_mask, deterministic=True)
    values = (
        action.selected_theta,
        action.log_prob_frontier,
        action.log_prob_theta,
        action.log_prob_total,
        output.value,
    )
    if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in values):
        raise Stage4MachineError("deterministic checkpoint action is not finite FP32")
    return {
        "selected_frontier_index": int(action.selected_frontier_index[0].cpu()),
        "selected_theta_fp32_hex": _tensor_fp32_hex(action.selected_theta[0]),
        "log_prob_frontier_fp32_hex": _tensor_fp32_hex(
            action.log_prob_frontier[0]
        ),
        "log_prob_theta_fp32_hex": _tensor_fp32_hex(action.log_prob_theta[0]),
        "log_prob_total_fp32_hex": _tensor_fp32_hex(action.log_prob_total[0]),
        "value_fp32_hex": _tensor_fp32_hex(output.value[0]),
    }


def _tensor_fp32_hex(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    return array.tobytes(order="C").hex()


def _action_fixture_npz(
    observation: PolicyObservation,
    action: dict[str, object],
) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(
        output,
        schema_json=np.asarray(
            json.dumps(
                {
                    "schema_version": STAGE4_ACTION_FIXTURE_SCHEMA_VERSION,
                    "observation_schema_version": PolicyObservation.schema_version,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        expected_action_json=np.asarray(
            json.dumps(action, sort_keys=True, separators=(",", ":"))
        ),
        prior_channels=observation.prior_channels,
        coverage_summary=observation.coverage_summary,
        local_crop=observation.local_crop,
        frontier_features=observation.frontier_features,
        pose_features=observation.pose_features,
        candidate_mask=observation.candidate_mask,
    )
    return output.getvalue()


def _ppo_metrics_record(metrics: object) -> dict[str, object]:
    return {
        "update_step": metrics.update_step,
        "initial_ratio_max_abs_error": metrics.initial_ratio_max_abs_error,
        "policy_loss": metrics.policy_loss,
        "value_loss": metrics.value_loss,
        "frontier_entropy_mean": metrics.frontier_entropy_mean,
        "frontier_entropy_min": metrics.frontier_entropy_min,
        "frontier_entropy_by_valid_candidate_count": {
            str(key): value
            for key, value in metrics.frontier_entropy_by_valid_candidate_count.items()
        },
        "theta_kappa_mean": metrics.theta_kappa_mean,
        "theta_kappa_max": metrics.theta_kappa_max,
        "approx_kl": metrics.approx_kl,
        "optimizer_steps": metrics.optimizer_steps,
        "planned_optimizer_steps": metrics.planned_optimizer_steps,
        "effective_minibatch_sizes": list(metrics.effective_minibatch_sizes),
        "physical_microbatch_sizes": list(metrics.physical_microbatch_sizes),
        "early_stopped": metrics.early_stopped,
        "early_stop_epoch": metrics.early_stop_epoch,
        "early_stop_minibatch": metrics.early_stop_minibatch,
        "skipped_minibatch_count": metrics.skipped_minibatch_count,
        "grad_pre_clip_norm_max": metrics.grad_pre_clip_norm_max,
        "grad_post_clip_norm_max": metrics.grad_post_clip_norm_max,
        "parameter_change_l2": metrics.parameter_change_l2,
        "policy_state_sha256_before": metrics.policy_state_sha256_before,
        "policy_state_sha256_after": metrics.policy_state_sha256_after,
    }


def _copy_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise Stage4MachineError("machine runtime state must be a string mapping")
    try:
        copied = json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise Stage4MachineError("machine runtime state must be finite JSON") from exc
    if not isinstance(copied, dict):
        raise Stage4MachineError("machine runtime state must be a JSON object")
    return copied


def _finite_tree(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _finite_tree(item) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(item) for item in value)
    return False


__all__ = [
    "STAGE4_ACTION_FIXTURE_SCHEMA_VERSION",
    "STAGE4_SMOKE_UPDATE_SCHEMA_VERSION",
    "Stage4MachineAuditBundle",
    "Stage4MachineError",
    "deterministic_action_record",
    "load_action_fixture_npz",
    "run_stage4_smoke_acceptance",
]
