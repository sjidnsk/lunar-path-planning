from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from lunar_exploration_ppo.ppo.checkpoint import (
    CheckpointError,
    CheckpointManager,
)
from lunar_exploration_ppo.ppo.trainer import policy_state_sha256


CONFIG_SHA = "a" * 64
LINEAGE = {
    "stage3_commit": "4df7be92cd6517e77c648e890bf7d32c9fcd560b",
    "stage3_gate_sha256": (
        "6ea3dc5bc199dd9370fab4b1a5aac1ebab08d9e42f49c0d01b3aa2b1e9aae0f0"
    ),
    "stage3_approval_sha256": (
        "8e901e04d587110de681da302ad24025dfebed945ff646ffa1e85bc5f5ecb3f2"
    ),
}
VERSIONS = {
    "observation_schema_version": "observed_map_frontier_tensor/v1",
    "action_space_version": "frontier_index_continuous_theta/v1",
    "network_architecture_version": "checkpoint_test_policy/v1",
    "reward_version": "stage1_reward/v1",
    "frontier_version": "stage1_frontier/v1",
    "planner_version": "stage1_planner/v1",
}


class _CheckpointPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4)
        self.theta = nn.Parameter(torch.tensor([0.25], dtype=torch.float32))

    def deterministic_action(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.linear(observation)
        return logits.argmax(dim=-1), self.theta.expand(observation.shape[0])


def _manager(tmp_path: Path, *, fault_phase: str | None = None) -> CheckpointManager:
    def fault_injector(phase: str) -> None:
        if phase == fault_phase:
            raise OSError(f"injected checkpoint failure: {phase}")

    return CheckpointManager(tmp_path / "checkpoints", fault_injector=fault_injector)


def _save(
    manager: CheckpointManager,
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    update_step: int,
    config_sha256: str = CONFIG_SHA,
    lineage: dict[str, str] = LINEAGE,
):
    return manager.save_complete(
        policy=policy,
        optimizer=optimizer,
        update_step=update_step,
        normalization_stats={"count": 17, "mean": [1.0, 2.0]},
        scenario_sampler_state={"seed": 91, "cursor": 7},
        vector_env_states=[
            {"worker": index, "episode": index + 10} for index in range(8)
        ],
        best_record={"success_rate": 0.25, "mean_final_coverage": 0.4},
        versions=VERSIONS,
        top_m_config={"top_m": 64},
        scale_profile="smoke/v1",
        training_config={"ppo_epochs": 4, "effective_minibatch_size": 256},
        config_sha256=config_sha256,
        lineage=lineage,
        eval_metrics={},
    )


def _policy_and_optimizer(
    seed: int = 7,
) -> tuple[_CheckpointPolicy, torch.optim.AdamW]:
    torch.manual_seed(seed)
    policy = _CheckpointPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3.0e-4)
    loss = policy.linear(torch.ones((2, 3), dtype=torch.float32)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return policy, optimizer


def test_complete_checkpoint_round_trip_restores_full_state_rng_and_action(
    tmp_path: Path,
) -> None:
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    policy, optimizer = _policy_and_optimizer(seed=17)
    observation = torch.tensor([[0.25, -0.5, 1.5]], dtype=torch.float32)
    action_before = tuple(
        value.clone() for value in policy.deterministic_action(observation)
    )
    hash_before = policy_state_sha256(policy)

    manager = _manager(tmp_path)
    receipt = _save(manager, policy=policy, optimizer=optimizer, update_step=1)
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)

    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(100.0)
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    loaded = manager.load_last_complete(
        policy=policy,
        optimizer=optimizer,
        expected_config_sha256=CONFIG_SHA,
        expected_lineage=LINEAGE,
    )

    assert receipt.update_step == 1
    assert receipt.checkpoint_path.drive.upper() == "D:"
    assert receipt.complete_marker_path.is_file()
    assert loaded.update_step == 1
    assert loaded.policy_state_sha256 == hash_before
    assert loaded.normalization_stats == {"count": 17, "mean": [1.0, 2.0]}
    assert loaded.scenario_sampler_state == {"seed": 91, "cursor": 7}
    assert loaded.vector_env_states == tuple(
        {"worker": index, "episode": index + 10} for index in range(8)
    )
    assert loaded.best_record == {
        "success_rate": 0.25,
        "mean_final_coverage": 0.4,
    }
    assert loaded.versions == VERSIONS
    assert loaded.training_config == {
        "ppo_epochs": 4,
        "effective_minibatch_size": 256,
    }
    assert loaded.eval_metrics == {}
    action_after = policy.deterministic_action(observation)
    assert torch.equal(action_after[0], action_before[0])
    assert torch.equal(action_after[1], action_before[1])
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)


@pytest.mark.parametrize(
    "fault_phase",
    (
        "after_payload_write",
        "after_manifest_write",
        "before_publish_rename",
    ),
)
def test_incomplete_save_is_invisible_and_loader_falls_back(
    tmp_path: Path,
    fault_phase: str,
) -> None:
    policy, optimizer = _policy_and_optimizer()
    stable = _manager(tmp_path)
    _save(stable, policy=policy, optimizer=optimizer, update_step=1)
    stable_hash = policy_state_sha256(policy)

    failing = _manager(tmp_path, fault_phase=fault_phase)
    with pytest.raises(OSError, match="injected checkpoint failure"):
        _save(failing, policy=policy, optimizer=optimizer, update_step=2)

    loaded = stable.load_last_complete(
        policy=policy,
        optimizer=optimizer,
        expected_config_sha256=CONFIG_SHA,
        expected_lineage=LINEAGE,
    )
    assert loaded.update_step == 1
    assert policy_state_sha256(policy) == stable_hash


def test_publish_fault_recovery_chain_keeps_atomic_update_and_advances(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer()
    stable = _manager(tmp_path)
    _save(stable, policy=policy, optimizer=optimizer, update_step=1)

    before_rename = _manager(tmp_path, fault_phase="before_publish_rename")
    with pytest.raises(OSError, match="before_publish_rename"):
        _save(before_rename, policy=policy, optimizer=optimizer, update_step=2)
    fallback = stable.load_last_complete(
        policy=policy,
        optimizer=optimizer,
        expected_config_sha256=CONFIG_SHA,
        expected_lineage=LINEAGE,
    )
    assert fallback.update_step == 1

    after_rename = _manager(tmp_path, fault_phase="after_publish_rename")
    with pytest.raises(OSError, match="after_publish_rename"):
        _save(after_rename, policy=policy, optimizer=optimizer, update_step=2)
    published = tmp_path / "checkpoints" / "update-00000002"
    assert (published / "checkpoint.pt").is_file()
    assert (published / "manifest.json").is_file()
    assert (published / "complete.json").is_file()

    recovered = stable.load_last_complete(
        policy=policy,
        optimizer=optimizer,
        expected_config_sha256=CONFIG_SHA,
        expected_lineage=LINEAGE,
    )
    assert recovered.update_step == 2

    third = _save(stable, policy=policy, optimizer=optimizer, update_step=3)
    advanced = stable.load_last_complete(
        policy=policy,
        optimizer=optimizer,
        expected_config_sha256=CONFIG_SHA,
        expected_lineage=LINEAGE,
    )
    assert advanced.update_step == 3
    assert advanced.checkpoint_sha256 == third.checkpoint_sha256
    with pytest.raises(CheckpointError, match="already exists"):
        _save(stable, policy=policy, optimizer=optimizer, update_step=3)


def test_corrupt_hash_or_missing_marker_falls_back_to_previous_complete(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer()
    manager = _manager(tmp_path)
    _save(manager, policy=policy, optimizer=optimizer, update_step=1)
    receipt = _save(manager, policy=policy, optimizer=optimizer, update_step=2)

    payload = receipt.checkpoint_path.read_bytes()
    receipt.checkpoint_path.write_bytes(
        payload[:-1] + bytes([payload[-1] ^ 0x01])
    )
    loaded = manager.load_last_complete(
        policy=policy,
        optimizer=optimizer,
        expected_config_sha256=CONFIG_SHA,
        expected_lineage=LINEAGE,
    )
    assert loaded.update_step == 1

    receipt3 = _save(manager, policy=policy, optimizer=optimizer, update_step=3)
    receipt3.complete_marker_path.unlink()
    loaded_again = manager.load_last_complete(
        policy=policy,
        optimizer=optimizer,
        expected_config_sha256=CONFIG_SHA,
        expected_lineage=LINEAGE,
    )
    assert loaded_again.update_step == 1


def test_lineage_and_config_mismatch_are_not_loadable(tmp_path: Path) -> None:
    policy, optimizer = _policy_and_optimizer()
    manager = _manager(tmp_path)
    _save(manager, policy=policy, optimizer=optimizer, update_step=1)
    _save(
        manager,
        policy=policy,
        optimizer=optimizer,
        update_step=2,
        config_sha256="b" * 64,
        lineage={**LINEAGE, "stage3_commit": "f" * 40},
    )

    loaded = manager.load_last_complete(
        policy=policy,
        optimizer=optimizer,
        expected_config_sha256=CONFIG_SHA,
        expected_lineage=LINEAGE,
    )
    assert loaded.update_step == 1
    with pytest.raises(CheckpointError, match="no compatible complete checkpoint"):
        manager.load_last_complete(
            policy=policy,
            optimizer=optimizer,
            expected_config_sha256="c" * 64,
            expected_lineage=LINEAGE,
        )


def test_completed_update_is_exclusive_and_cannot_be_overwritten(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer()
    manager = _manager(tmp_path)
    first = _save(manager, policy=policy, optimizer=optimizer, update_step=1)
    first_bytes = first.checkpoint_path.read_bytes()

    with pytest.raises(CheckpointError, match="already exists"):
        _save(manager, policy=policy, optimizer=optimizer, update_step=1)

    assert first.checkpoint_path.read_bytes() == first_bytes


def test_checkpoint_rejects_non_d_root_and_incomplete_metadata(
    tmp_path: Path,
) -> None:
    if Path(tmp_path).drive.upper() != "D:":
        with pytest.raises(CheckpointError, match="D drive"):
            CheckpointManager(tmp_path / "checkpoints")
    policy, optimizer = _policy_and_optimizer()
    manager = _manager(tmp_path)
    with pytest.raises(CheckpointError, match="exactly 8 vector env states"):
        manager.save_complete(
            policy=policy,
            optimizer=optimizer,
            update_step=1,
            normalization_stats={},
            scenario_sampler_state={},
            vector_env_states=[],
            best_record={},
            versions=VERSIONS,
            top_m_config={"top_m": 64},
            scale_profile="smoke/v1",
            training_config={},
            config_sha256=CONFIG_SHA,
            lineage=LINEAGE,
            eval_metrics={},
        )
