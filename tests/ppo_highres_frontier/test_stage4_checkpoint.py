from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from lunar_exploration_ppo.configs.stage6 import (
    SAFETY_CONTRACT_SOURCE,
    SafetyContract,
    load_stage6_config,
)
from lunar_exploration_ppo.ppo.checkpoint import (
    CheckpointError,
    CheckpointManager,
    STAGE6_CHECKPOINT_SCHEMA_VERSION,
)
from lunar_exploration_ppo.ppo.checkpoint_retention import (
    CheckpointRetentionError,
    CheckpointRetentionManager,
)
from lunar_exploration_ppo.ppo.trainer import policy_state_sha256
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
import lunar_exploration_ppo.ppo.checkpoint as checkpoint_module


CONFIG_SHA = "a" * 64
ROOT = Path(__file__).resolve().parents[2]
STAGE6_CONFIG_PATH = ROOT / "configs/ppo_highres_frontier_stage6_v1.json"
STAGE6_CONFIG_SHA = hashlib.sha256(STAGE6_CONFIG_PATH.read_bytes()).hexdigest()
STAGE6_SAFETY_CONTRACT = SafetyContract.from_stage6_config(
    load_stage6_config(STAGE6_CONFIG_PATH)
)
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
STAGE6_LINEAGE = {
    "stage5_commit": "b" * 40,
    "stage5_gate_sha256": "1" * 64,
    "stage5_manifest_sha256": "2" * 64,
    "stage4_checkpoint_sha256": "3" * 64,
    "stage4_policy_state_sha256": "4" * 64,
    "source_set_sha256": "5" * 64,
    "prospective_tree_sha256": "6" * 64,
    "data_sha256": "7" * 64,
    "safety_contract_sha256": STAGE6_SAFETY_CONTRACT.sha256,
}
SAFETY_CONTRACT = STAGE6_SAFETY_CONTRACT.to_dict()


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


def _stage6_manager(
    tmp_path: Path,
    *,
    fault_injector=None,
) -> CheckpointManager:
    return CheckpointManager(
        tmp_path / "stage6-checkpoints",
        fault_injector=fault_injector,
        schema_version=STAGE6_CHECKPOINT_SCHEMA_VERSION,
    )


def _stage4_checkpoint_kwargs(
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    update_step: int,
    config_sha256: str = CONFIG_SHA,
    lineage: dict[str, str] = LINEAGE,
) -> dict[str, object]:
    return {
        "policy": policy,
        "optimizer": optimizer,
        "update_step": update_step,
        "normalization_stats": {"count": 17, "mean": [1.0, 2.0]},
        "scenario_sampler_state": {"seed": 91, "cursor": 7},
        "vector_env_states": [
            {"worker": index, "episode": index + 10} for index in range(8)
        ],
        "best_record": {"success_rate": 0.25, "mean_final_coverage": 0.4},
        "versions": VERSIONS,
        "top_m_config": {"top_m": 64},
        "scale_profile": "smoke/v1",
        "training_config": {"ppo_epochs": 4, "effective_minibatch_size": 256},
        "config_sha256": config_sha256,
        "lineage": lineage,
        "eval_metrics": {},
    }


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
        **_stage4_checkpoint_kwargs(
            policy=policy,
            optimizer=optimizer,
            update_step=update_step,
            config_sha256=config_sha256,
            lineage=lineage,
        )
    )


def _save_stage6(
    manager: CheckpointManager,
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    update_step: int,
    safety_contract: SafetyContract | None = STAGE6_SAFETY_CONTRACT,
):
    return manager.save_complete(
        policy=policy,
        optimizer=optimizer,
        update_step=update_step,
        normalization_stats={},
        scenario_sampler_state={"workers": list(range(8))},
        vector_env_states=[{"worker": index} for index in range(8)],
        best_record={},
        versions=VERSIONS,
        top_m_config={"top_m": 1024},
        scale_profile="Standard v1",
        training_config={"updates_per_seed": 100},
        config_sha256=STAGE6_CONFIG_SHA,
        lineage=STAGE6_LINEAGE,
        eval_metrics={},
        safety_contract=safety_contract,
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


def _make_directory_reparse(link: Path, target: Path) -> None:
    if os.name != "nt":
        os.symlink(target, link, target_is_directory=True)
        return
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            "Windows junction creation failed: "
            f"{completed.stdout} {completed.stderr}"
        )


def _inject_nonfinite_state(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    state_kind: str,
    value: float,
) -> None:
    if state_kind == "model":
        with torch.no_grad():
            next(policy.parameters()).reshape(-1)[0] = value
        return

    if state_kind != "optimizer":
        raise AssertionError(f"unsupported state kind: {state_kind}")
    optimizer_state = next(
        state
        for state in optimizer.state.values()
        if any(
            isinstance(item, torch.Tensor)
            and (item.is_floating_point() or item.is_complex())
            and item.numel() > 0
            for item in state.values()
        )
    )
    tensor_name = next(
        name
        for name, item in optimizer_state.items()
        if isinstance(item, torch.Tensor)
        and (item.is_floating_point() or item.is_complex())
        and item.numel() > 0
    )
    corrupted = optimizer_state[tensor_name].detach().clone()
    corrupted.reshape(-1)[0] = value
    optimizer_state[tensor_name] = corrupted


def _rewrite_checkpoint_payload(receipt, payload: dict[str, object]) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    payload_bytes = buffer.getvalue()
    receipt.checkpoint_path.write_bytes(payload_bytes)

    manifest = json.loads(receipt.manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint"]["sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    manifest["checkpoint"]["size_bytes"] = len(payload_bytes)
    manifest["policy_state_sha256"] = payload["policy_state_sha256"]
    manifest_bytes = ArtifactStore.canonical_json_bytes(manifest)
    receipt.manifest_path.write_bytes(manifest_bytes)

    complete = json.loads(
        receipt.complete_marker_path.read_text(encoding="utf-8")
    )
    complete["checkpoint_sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    complete["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    receipt.complete_marker_path.write_bytes(
        ArtifactStore.canonical_json_bytes(complete)
    )


@pytest.mark.parametrize(
    ("state_kind", "value"),
    [
        pytest.param("model", float("nan"), id="model-nan"),
        pytest.param("optimizer", float("inf"), id="optimizer-positive-inf"),
        pytest.param("optimizer", float("-inf"), id="optimizer-negative-inf"),
    ],
)
def test_checkpoint_save_rejects_nonfinite_state_without_artifacts(
    tmp_path: Path,
    state_kind: str,
    value: float,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=61)
    _inject_nonfinite_state(
        policy,
        optimizer,
        state_kind=state_kind,
        value=value,
    )
    manager = _manager(tmp_path)
    root_members_before = tuple(manager.root.iterdir())

    with pytest.raises(CheckpointError) as captured:
        _save(manager, policy=policy, optimizer=optimizer, update_step=1)

    assert str(captured.value) == (
        f"checkpoint {state_kind} state contains non-finite tensor"
    )
    assert tuple(manager.root.iterdir()) == root_members_before == ()


@pytest.mark.parametrize(
    ("state_kind", "value"),
    [
        pytest.param("model", float("nan"), id="model-nan"),
        pytest.param("optimizer", float("inf"), id="optimizer-positive-inf"),
        pytest.param("optimizer", float("-inf"), id="optimizer-negative-inf"),
    ],
)
def test_checkpoint_load_rejects_rehashed_nonfinite_state_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state_kind: str,
    value: float,
) -> None:
    source_policy, source_optimizer = _policy_and_optimizer(seed=67)
    manager = _manager(tmp_path)
    receipt = _save(
        manager,
        policy=source_policy,
        optimizer=source_optimizer,
        update_step=1,
    )
    payload = torch.load(
        receipt.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if state_kind == "model":
        model_tensor = next(
            item
            for item in payload["model_state_dict"].values()
            if isinstance(item, torch.Tensor)
            and (item.is_floating_point() or item.is_complex())
            and item.numel() > 0
        )
        model_tensor.reshape(-1)[0] = value
        rebound_policy = _CheckpointPolicy()
        rebound_policy.load_state_dict(payload["model_state_dict"], strict=True)
        payload["policy_state_sha256"] = policy_state_sha256(rebound_policy)
    else:
        optimizer_state = next(
            state
            for state in payload["optimizer_state_dict"]["state"].values()
            if any(
                isinstance(item, torch.Tensor)
                and (item.is_floating_point() or item.is_complex())
                and item.numel() > 0
                for item in state.values()
            )
        )
        optimizer_tensor = next(
            item
            for item in optimizer_state.values()
            if isinstance(item, torch.Tensor)
            and (item.is_floating_point() or item.is_complex())
            and item.numel() > 0
        )
        optimizer_tensor.reshape(-1)[0] = value
    _rewrite_checkpoint_payload(receipt, payload)

    target_policy, target_optimizer = _policy_and_optimizer(seed=71)
    policy_before = copy.deepcopy(target_policy.state_dict())
    optimizer_before = copy.deepcopy(target_optimizer.state_dict())
    load_calls = {"policy": 0, "optimizer": 0}
    policy_load_state_dict = target_policy.load_state_dict
    optimizer_load_state_dict = target_optimizer.load_state_dict

    def tracked_policy_load(*args, **kwargs):
        load_calls["policy"] += 1
        return policy_load_state_dict(*args, **kwargs)

    def tracked_optimizer_load(*args, **kwargs):
        load_calls["optimizer"] += 1
        return optimizer_load_state_dict(*args, **kwargs)

    monkeypatch.setattr(target_policy, "load_state_dict", tracked_policy_load)
    monkeypatch.setattr(target_optimizer, "load_state_dict", tracked_optimizer_load)

    with pytest.raises(CheckpointError) as captured:
        manager.load_complete(
            update_step=1,
            policy=target_policy,
            optimizer=target_optimizer,
            expected_config_sha256=CONFIG_SHA,
            expected_lineage=LINEAGE,
        )

    assert str(captured.value) == (
        f"checkpoint {state_kind} state contains non-finite tensor"
    )
    assert load_calls == {"policy": 0, "optimizer": 0}
    torch.testing.assert_close(
        target_policy.state_dict(),
        policy_before,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        target_optimizer.state_dict(),
        optimizer_before,
        rtol=0,
        atol=0,
    )


def test_checkpoint_finite_validation_accepts_integer_and_bool_state_tensors(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=73)
    policy.register_buffer("integer_counter", torch.tensor(3, dtype=torch.int64))
    policy.register_buffer(
        "boolean_mask",
        torch.tensor([True, False], dtype=torch.bool),
    )
    manager = _manager(tmp_path)

    receipt = _save(manager, policy=policy, optimizer=optimizer, update_step=1)
    inspected = manager.inspect_complete(
        update_step=1,
        expected_config_sha256=CONFIG_SHA,
        expected_lineage=LINEAGE,
    )

    assert inspected.checkpoint_sha256 == receipt.checkpoint_sha256


def test_checkpoint_rejects_nonfinite_metadata_on_write_and_read(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer()
    manager = _manager(tmp_path)
    arguments = _stage4_checkpoint_kwargs(
        policy=policy,
        optimizer=optimizer,
        update_step=1,
    )
    arguments["eval_metrics"] = {"nested": [{"loss": float("nan")}]}

    with pytest.raises(CheckpointError, match="canonical JSON"):
        manager.save_complete(**arguments)

    checkpoint_root = tmp_path / "checkpoints"
    assert not list(checkpoint_root.glob("update-*"))

    forged = tmp_path / "forged-manifest.json"
    forged.write_bytes(b'{\n  "value": NaN\n}\n')
    with pytest.raises(CheckpointError, match="invalid JSON|not canonical"):
        checkpoint_module._read_canonical_json(forged)


def test_checkpoint_manager_rejects_root_reparse_before_store_access(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-checkpoint-root"
    outside.mkdir()
    linked = tmp_path / "linked-checkpoint-root"
    _make_directory_reparse(linked, outside)

    with pytest.raises(CheckpointError, match="link|reparse"):
        CheckpointManager(linked)


def test_checkpoint_manager_rejects_update_reparse_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=13)
    manager = _manager(tmp_path)
    receipt = _save(
        manager,
        policy=policy,
        optimizer=optimizer,
        update_step=1,
    )
    outside = tmp_path / "outside-update"
    receipt.directory.rename(outside)
    _make_directory_reparse(receipt.directory, outside)

    def reject_decode(*args: object, **kwargs: object) -> object:
        raise AssertionError("torch.load reached through update reparse")

    monkeypatch.setattr(checkpoint_module.torch, "load", reject_decode)
    with pytest.raises(CheckpointError, match="link|reparse"):
        manager.inspect_complete(
            update_step=1,
            expected_config_sha256=CONFIG_SHA,
            expected_lineage=LINEAGE,
        )


def test_stage6_checkpoint_manifest_and_payload_bind_explicit_safety_contract(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=14)
    manager = _stage6_manager(tmp_path)
    receipt = _save_stage6(
        manager,
        policy=policy,
        optimizer=optimizer,
        update_step=1,
    )

    manifest = json.loads(receipt.manifest_path.read_text(encoding="utf-8"))
    payload = torch.load(
        receipt.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert manifest["safety_contract"] == SAFETY_CONTRACT
    assert manifest["safety_contract_sha256"] == hashlib.sha256(
        ArtifactStore.canonical_json_bytes(SAFETY_CONTRACT)
    ).hexdigest()
    assert payload["safety_contract"] == SAFETY_CONTRACT
    assert manifest["safety_contract_source"] == SAFETY_CONTRACT_SOURCE
    assert manifest["safety_contract_config_sha256"] == STAGE6_CONFIG_SHA
    assert payload["safety_contract_source"] == SAFETY_CONTRACT_SOURCE
    assert payload["safety_contract_config_sha256"] == STAGE6_CONFIG_SHA


def test_bound_checkpoint_snapshot_inspects_restores_and_rejects_membership(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=141)
    receipt = _save_stage6(
        _stage6_manager(tmp_path),
        policy=policy,
        optimizer=optimizer,
        update_step=1,
    )
    snapshot = {
        "directory": receipt.directory,
        "member_names": ("checkpoint.pt", "complete.json", "manifest.json"),
        "checkpoint_bytes": receipt.checkpoint_path.read_bytes(),
        "manifest_bytes": receipt.manifest_path.read_bytes(),
        "complete_bytes": receipt.complete_marker_path.read_bytes(),
    }

    inspected = checkpoint_module.inspect_complete_checkpoint_snapshot(
        **snapshot,
        update_step=1,
        schema_version=STAGE6_CHECKPOINT_SCHEMA_VERSION,
        expected_config_sha256=STAGE6_CONFIG_SHA,
        expected_lineage=STAGE6_LINEAGE,
        expected_safety_contract=STAGE6_SAFETY_CONTRACT,
    )
    assert inspected.checkpoint_sha256 == receipt.checkpoint_sha256
    assert inspected.policy_state_sha256 == receipt.policy_state_sha256

    restored_policy, restored_optimizer = _policy_and_optimizer(seed=142)
    loaded = checkpoint_module.load_complete_checkpoint_snapshot(
        **snapshot,
        update_step=1,
        schema_version=STAGE6_CHECKPOINT_SCHEMA_VERSION,
        policy=restored_policy,
        optimizer=restored_optimizer,
        expected_config_sha256=STAGE6_CONFIG_SHA,
        expected_lineage=STAGE6_LINEAGE,
        expected_safety_contract=STAGE6_SAFETY_CONTRACT,
    )
    assert loaded.checkpoint_sha256 == receipt.checkpoint_sha256
    assert policy_state_sha256(restored_policy) == receipt.policy_state_sha256

    with pytest.raises(CheckpointError, match="membership"):
        checkpoint_module.inspect_complete_checkpoint_snapshot(
            **{**snapshot, "member_names": ("checkpoint.pt", "manifest.json")},
            update_step=1,
            schema_version=STAGE6_CHECKPOINT_SCHEMA_VERSION,
            expected_config_sha256=STAGE6_CONFIG_SHA,
            expected_lineage=STAGE6_LINEAGE,
            expected_safety_contract=STAGE6_SAFETY_CONTRACT,
        )


def test_bound_retention_snapshot_rejects_unknown_directory_members(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=143)
    receipt = _save_stage6(
        _stage6_manager(tmp_path),
        policy=policy,
        optimizer=optimizer,
        update_step=1,
    )
    identity = {
        "device": 0,
        "inode": 0,
        "mode": 0,
        "file_attributes": 0,
        "link_count": 1,
    }
    directories = [
        {
            "path": "checkpoints/seed-20260716",
            "identity": identity,
            "members": [
                {"name": "latest.json", "kind": "file"},
                {"name": "update-00000001", "kind": "directory"},
            ],
        },
        {
            "path": "checkpoints/seed-20260716/update-00000001",
            "identity": identity,
            "members": [
                {"name": "checkpoint.pt", "kind": "file"},
                {"name": "complete.json", "kind": "file"},
                {"name": "manifest.json", "kind": "file"},
            ],
        },
    ]
    payloads = {
        "checkpoints/seed-20260716/latest.json": (
            receipt.directory.parent / "latest.json"
        ).read_bytes(),
        "checkpoints/seed-20260716/update-00000001/complete.json": (
            receipt.complete_marker_path.read_bytes()
        ),
    }

    def read_bound(relative: str, *, label: str) -> bytes:
        assert label
        return payloads[relative]

    from lunar_exploration_ppo.ppo import checkpoint_retention as retention_module

    verified = retention_module.verify_retained_checkpoint_snapshot(
        root_relative="checkpoints/seed-20260716",
        schema_version=STAGE6_CHECKPOINT_SCHEMA_VERSION,
        latest_update=1,
        expected_updates=(1,),
        directory_snapshot=directories,
        read_bytes=read_bound,
    )
    assert verified.kept_updates == (1,)

    drifted = copy.deepcopy(directories)
    drifted[1]["members"].append({"name": "rogue.bin", "kind": "file"})
    with pytest.raises(CheckpointRetentionError, match="member"):
        retention_module.verify_retained_checkpoint_snapshot(
            root_relative="checkpoints/seed-20260716",
            schema_version=STAGE6_CHECKPOINT_SCHEMA_VERSION,
            latest_update=1,
            expected_updates=(1,),
            directory_snapshot=drifted,
            read_bytes=read_bound,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vehicle_radius_m", 0.4215874762),
        ("safety_margin_m", 0.11),
        ("min_clearance_m", 0.5215874762),
        ("traversability_threshold", 0.51),
        ("max_traversable_slope_deg", 29.0),
    ],
)
def test_stage6_checkpoint_rejects_rehashed_manifest_safety_component_drift(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=15)
    manager = _stage6_manager(tmp_path)
    receipt = _save_stage6(
        manager,
        policy=policy,
        optimizer=optimizer,
        update_step=1,
    )
    manifest = json.loads(receipt.manifest_path.read_text(encoding="utf-8"))
    manifest["safety_contract"] = {
        **SAFETY_CONTRACT,
        field: value,
    }
    manifest["safety_contract_sha256"] = hashlib.sha256(
        ArtifactStore.canonical_json_bytes(manifest["safety_contract"])
    ).hexdigest()
    manifest_bytes = ArtifactStore.canonical_json_bytes(manifest)
    receipt.manifest_path.write_bytes(manifest_bytes)
    complete = json.loads(
        receipt.complete_marker_path.read_text(encoding="utf-8")
    )
    complete["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    receipt.complete_marker_path.write_bytes(
        ArtifactStore.canonical_json_bytes(complete)
    )

    with pytest.raises(CheckpointError, match="safety"):
        manager.inspect_complete(
            update_step=1,
            expected_config_sha256=STAGE6_CONFIG_SHA,
            expected_lineage=STAGE6_LINEAGE,
            expected_safety_contract=STAGE6_SAFETY_CONTRACT,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vehicle_radius_m", 0.4215874762),
        ("safety_margin_m", 0.11),
        ("min_clearance_m", 0.5215874762),
        ("traversability_threshold", 0.51),
        ("max_traversable_slope_deg", 29.0),
    ],
)
def test_stage6_checkpoint_rejects_rehashed_payload_safety_component_drift(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=16)
    manager = _stage6_manager(tmp_path)
    receipt = _save_stage6(
        manager,
        policy=policy,
        optimizer=optimizer,
        update_step=1,
    )
    payload = torch.load(
        receipt.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    payload["safety_contract"] = {
        **SAFETY_CONTRACT,
        field: value,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    payload_bytes = buffer.getvalue()
    receipt.checkpoint_path.write_bytes(payload_bytes)
    manifest = json.loads(receipt.manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint"]["sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    manifest["checkpoint"]["size_bytes"] = len(payload_bytes)
    manifest_bytes = ArtifactStore.canonical_json_bytes(manifest)
    receipt.manifest_path.write_bytes(manifest_bytes)
    complete = json.loads(
        receipt.complete_marker_path.read_text(encoding="utf-8")
    )
    complete["checkpoint_sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    complete["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    receipt.complete_marker_path.write_bytes(
        ArtifactStore.canonical_json_bytes(complete)
    )

    with pytest.raises(CheckpointError, match="safety"):
        manager.inspect_complete(
            update_step=1,
            expected_config_sha256=STAGE6_CONFIG_SHA,
            expected_lineage=STAGE6_LINEAGE,
            expected_safety_contract=STAGE6_SAFETY_CONTRACT,
        )


def test_checkpoint_profiles_fail_safe_on_missing_or_cross_stage_safety_contract(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=17)

    with pytest.raises(CheckpointError, match="safety"):
        _save_stage6(
            _stage6_manager(tmp_path),
            policy=policy,
            optimizer=optimizer,
            update_step=1,
            safety_contract=None,
        )

    with pytest.raises(CheckpointError, match="safety"):
        _manager(tmp_path).save_complete(
            policy=policy,
            optimizer=optimizer,
            update_step=1,
            normalization_stats={},
            scenario_sampler_state={},
            vector_env_states=[{"worker": index} for index in range(8)],
            best_record={},
            versions=VERSIONS,
            top_m_config={"top_m": 64},
            scale_profile="smoke/v1",
            training_config={},
            config_sha256=CONFIG_SHA,
            lineage=LINEAGE,
            eval_metrics={},
            safety_contract=STAGE6_SAFETY_CONTRACT,
        )


def test_stage6_checkpoint_restore_requires_matching_safety_config_source(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=18)
    manager = _stage6_manager(tmp_path)
    _save_stage6(
        manager,
        policy=policy,
        optimizer=optimizer,
        update_step=1,
    )

    restored_policy, restored_optimizer = _policy_and_optimizer(seed=19)
    with pytest.raises(CheckpointError, match="safety"):
        manager.load_complete(
            update_step=1,
            policy=restored_policy,
            optimizer=restored_optimizer,
            expected_config_sha256=STAGE6_CONFIG_SHA,
            expected_lineage=STAGE6_LINEAGE,
        )


def test_checkpoint_save_rejects_leaf_replacement_after_publish(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage6-checkpoints"

    def replace_leaf(phase: str) -> None:
        if phase == "after_publish_rename":
            leaf = root / "update-00000001/checkpoint.pt"
            leaf.replace(tmp_path / "displaced-checkpoint.pt")
            leaf.write_bytes(b"raced")

    policy, optimizer = _policy_and_optimizer(seed=18)
    manager = _stage6_manager(tmp_path, fault_injector=replace_leaf)

    with pytest.raises(CheckpointError, match="identity changed"):
        _save_stage6(
            manager,
            policy=policy,
            optimizer=optimizer,
            update_step=1,
        )


def test_checkpoint_save_rejects_parent_replacement_after_publish(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage6-checkpoints"

    def replace_parent(phase: str) -> None:
        if phase == "after_publish_rename":
            directory = root / "update-00000001"
            directory.replace(tmp_path / "displaced-update")
            directory.mkdir()

    policy, optimizer = _policy_and_optimizer(seed=19)
    manager = _stage6_manager(tmp_path, fault_injector=replace_parent)

    with pytest.raises(CheckpointError, match="identity changed"):
        _save_stage6(
            manager,
            policy=policy,
            optimizer=optimizer,
            update_step=1,
        )


def test_checkpoint_read_rejects_leaf_replacement_after_descriptor_read(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=20)
    writer = _stage6_manager(tmp_path)
    receipt = _save_stage6(
        writer,
        policy=policy,
        optimizer=optimizer,
        update_step=1,
    )

    def replace_leaf(phase: str) -> None:
        if phase == "after_checkpoint_read":
            receipt.checkpoint_path.replace(tmp_path / "read-displaced.pt")
            receipt.checkpoint_path.write_bytes(b"raced")

    reader = _stage6_manager(tmp_path, fault_injector=replace_leaf)
    with pytest.raises(CheckpointError, match="identity changed"):
        reader.inspect_complete(
            update_step=1,
            expected_config_sha256=STAGE6_CONFIG_SHA,
            expected_lineage=STAGE6_LINEAGE,
            expected_safety_contract=STAGE6_SAFETY_CONTRACT,
        )


def test_checkpoint_read_rejects_parent_replacement_during_member_reads(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=21)
    writer = _stage6_manager(tmp_path)
    receipt = _save_stage6(
        writer,
        policy=policy,
        optimizer=optimizer,
        update_step=1,
    )

    def replace_parent(phase: str) -> None:
        if phase == "after_manifest_read":
            displaced = tmp_path / "read-displaced-update"
            receipt.directory.replace(displaced)
            receipt.directory.mkdir()
            for name in ("checkpoint.pt", "manifest.json", "complete.json"):
                shutil.copy2(displaced / name, receipt.directory / name)

    reader = _stage6_manager(tmp_path, fault_injector=replace_parent)
    with pytest.raises(CheckpointError, match="identity changed"):
        reader.inspect_complete(
            update_step=1,
            expected_config_sha256=STAGE6_CONFIG_SHA,
            expected_lineage=STAGE6_LINEAGE,
            expected_safety_contract=STAGE6_SAFETY_CONTRACT,
        )


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW unavailable")
def test_checkpoint_member_reads_request_no_follow_when_platform_supports_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=22)
    manager = _manager(tmp_path)
    _save(manager, policy=policy, optimizer=optimizer, update_step=1)
    observed_flags: list[int] = []
    real_open = checkpoint_module.os.open

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):
        observed_flags.append(int(flags))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(checkpoint_module.os, "open", recording_open)
    manager.inspect_complete(
        update_step=1,
        expected_config_sha256=CONFIG_SHA,
        expected_lineage=LINEAGE,
    )

    assert observed_flags
    assert all(flags & os.O_NOFOLLOW for flags in observed_flags)


def _two_update_checkpoint_root(tmp_path: Path) -> Path:
    policy, optimizer = _policy_and_optimizer(seed=23)
    manager = _manager(tmp_path)
    _save(manager, policy=policy, optimizer=optimizer, update_step=1)
    _save(manager, policy=policy, optimizer=optimizer, update_step=2)
    return manager.root


def test_checkpoint_retention_rejects_leaf_replacement_before_unlink(
    tmp_path: Path,
) -> None:
    root = _two_update_checkpoint_root(tmp_path)

    def replace_leaf(phase: str) -> None:
        if phase == "before_member_unlink:checkpoint.pt":
            leaf = root / "update-00000001/checkpoint.pt"
            leaf.replace(tmp_path / "retention-displaced.pt")
            leaf.write_bytes(b"raced")

    manager = CheckpointRetentionManager(root, fault_injector=replace_leaf)
    with pytest.raises(CheckpointRetentionError, match="identity changed"):
        manager.apply(
            latest_update=2,
            periodic_updates=(2,),
            best_update=2,
            periodic_keep_count=5,
        )


def test_checkpoint_retention_rejects_parent_replacement_before_rmdir(
    tmp_path: Path,
) -> None:
    root = _two_update_checkpoint_root(tmp_path)

    def replace_parent(phase: str) -> None:
        if phase == "before_directory_rmdir":
            directory = root / "update-00000001"
            directory.replace(tmp_path / "retention-displaced-update")
            directory.mkdir()

    manager = CheckpointRetentionManager(root, fault_injector=replace_parent)
    with pytest.raises(CheckpointRetentionError, match="identity changed"):
        manager.apply(
            latest_update=2,
            periodic_updates=(2,),
            best_update=2,
            periodic_keep_count=5,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-bound namespace contract")
def test_checkpoint_retention_update_rename_binds_validated_directory_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.utils.path_security as path_security

    root = _two_update_checkpoint_root(tmp_path)
    source = root / "update-00000001"
    displaced = tmp_path / "retention-handle-bound-update"
    boundary_observed = False
    replacement_blocked = False

    def attempt_replacement(
        event: str,
        event_source: Path,
        event_destination: Path | None,
    ) -> None:
        nonlocal boundary_observed, replacement_blocked
        if (
            event != "after_source_handle_bound"
            or event_source != source
            or event_destination is None
        ):
            return
        boundary_observed = True
        try:
            os.replace(event_source, displaced)
        except OSError:
            replacement_blocked = True

    monkeypatch.setattr(
        path_security,
        "_namespace_event",
        attempt_replacement,
        raising=False,
    )

    receipt = CheckpointRetentionManager(root).apply(
        latest_update=2,
        periodic_updates=(2,),
        best_update=2,
        periodic_keep_count=5,
    )

    assert receipt.removed_updates == (1,)
    assert boundary_observed is True
    assert replacement_blocked is True
    assert not source.exists()
    assert not displaced.exists()


def test_artifact_store_atomic_and_exclusive_use_durable_namespace_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.utils.artifact_io as artifact_module

    events: list[tuple[str, str, str]] = []

    def durable_replace(
        source: Path,
        destination: Path,
        *,
        expected_source_identity: tuple[int, int, int, int, int, int],
        replace_existing: bool = True,
    ) -> None:
        events.append(("replace", Path(source).name, Path(destination).name))
        assert checkpoint_module._regular_path_identity(Path(source)) == (
            expected_source_identity
        )
        assert replace_existing is True
        os.replace(source, destination)

    def durable_publish_exclusive(
        source: Path,
        destination: Path,
        *,
        expected_source_identity: tuple[int, int, int, int, int, int],
    ) -> None:
        events.append(("exclusive", Path(source).name, Path(destination).name))
        assert checkpoint_module._regular_path_identity(Path(source)) == (
            expected_source_identity
        )
        os.link(source, destination)
        Path(source).unlink()

    monkeypatch.setattr(
        artifact_module,
        "durable_replace",
        durable_replace,
        raising=False,
    )
    monkeypatch.setattr(
        artifact_module,
        "durable_publish_exclusive",
        durable_publish_exclusive,
        raising=False,
    )
    store = artifact_module.ArtifactStore(tmp_path)

    store.write_bytes("atomic.bin", b"atomic")
    store.write_bytes_exclusive("exclusive.bin", b"exclusive")

    assert [event[0] for event in events] == ["replace", "exclusive"]
    assert events[0][2] == "atomic.bin"
    assert events[1][2] == "exclusive.bin"
    assert (tmp_path / "atomic.bin").read_bytes() == b"atomic"
    assert (tmp_path / "exclusive.bin").read_bytes() == b"exclusive"


@pytest.mark.parametrize(
    ("exclusive", "primitive_name"),
    ((False, "durable_replace"), (True, "durable_publish_exclusive")),
)
def test_artifact_store_fails_closed_when_namespace_durability_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exclusive: bool,
    primitive_name: str,
) -> None:
    import lunar_exploration_ppo.utils.artifact_io as artifact_module

    failure = OSError("injected durable namespace failure")

    def fail_closed(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(
        artifact_module,
        primitive_name,
        fail_closed,
        raising=False,
    )
    store = artifact_module.ArtifactStore(tmp_path)
    target = tmp_path / ("exclusive.bin" if exclusive else "atomic.bin")

    with pytest.raises(OSError) as captured:
        if exclusive:
            store.write_bytes_exclusive(target.name, b"payload")
        else:
            store.write_bytes(target.name, b"payload")

    assert captured.value is failure
    assert not target.exists()


def test_artifact_store_failed_publish_cleanup_preserves_replaced_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.utils.artifact_io as artifact_module

    primary = OSError("injected publication failure")
    foreign = b"foreign-temporary-bytes"
    owned = b"owned-temporary-bytes"
    displaced = tmp_path / "displaced-owned-temporary.bin"
    captured_temporary: Path | None = None

    def fail_after_replacing_temporary(
        source: Path,
        _destination: Path,
        **_kwargs: object,
    ) -> None:
        nonlocal captured_temporary
        captured_temporary = Path(source)
        os.replace(source, displaced)
        Path(source).write_bytes(foreign)
        raise primary

    monkeypatch.setattr(
        artifact_module,
        "durable_replace",
        fail_after_replacing_temporary,
    )

    with pytest.raises(OSError) as captured:
        artifact_module.ArtifactStore(tmp_path).write_bytes("target.bin", owned)

    assert captured.value is primary
    assert captured_temporary is not None
    assert captured_temporary.read_bytes() == foreign
    assert displaced.read_bytes() == owned
    assert not (tmp_path / "target.bin").exists()


def test_artifact_store_failed_exclusive_cleanup_preserves_replaced_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.utils.artifact_io as artifact_module

    primary = OSError("injected exclusive publication failure")
    foreign = b"foreign-exclusive-temporary"
    owned = b"owned-exclusive-temporary"
    displaced = tmp_path / "displaced-owned-exclusive.bin"
    captured_temporary: Path | None = None

    def fail_after_replacing_temporary(
        source: Path,
        _destination: Path,
        **_kwargs: object,
    ) -> None:
        nonlocal captured_temporary
        captured_temporary = Path(source)
        os.replace(source, displaced)
        Path(source).write_bytes(foreign)
        raise primary

    monkeypatch.setattr(
        artifact_module,
        "durable_publish_exclusive",
        fail_after_replacing_temporary,
    )

    with pytest.raises(OSError) as captured:
        artifact_module.ArtifactStore(tmp_path).write_bytes_exclusive(
            "exclusive-target.bin",
            owned,
        )

    assert captured.value is primary
    assert captured_temporary is not None
    assert captured_temporary.read_bytes() == foreign
    assert displaced.read_bytes() == owned
    assert not (tmp_path / "exclusive-target.bin").exists()


@pytest.mark.parametrize("close_fails", (False, True))
def test_artifact_store_write_failure_cleans_bound_temporary_without_masking_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    close_fails: bool,
) -> None:
    import lunar_exploration_ppo.utils.artifact_io as artifact_module

    primary = OSError("injected descriptor write failure")
    secondary = OSError("injected descriptor close failure")
    captured_temporary: Path | None = None
    target_descriptor: int | None = None
    real_close = artifact_module.os.close

    def fail_write(
        descriptor: int,
        path: Path,
        _payload: bytes,
        *,
        parent_guard: object,
    ) -> None:
        nonlocal captured_temporary, target_descriptor
        del parent_guard
        captured_temporary = path
        target_descriptor = descriptor
        raise primary

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        if close_fails and descriptor == target_descriptor:
            raise secondary

    monkeypatch.setattr(
        artifact_module.ArtifactStore,
        "_write_all",
        staticmethod(fail_write),
    )
    monkeypatch.setattr(artifact_module.os, "close", close_then_fail)

    with pytest.raises(OSError) as captured:
        artifact_module.ArtifactStore(tmp_path).write_bytes(
            "write-failure.bin",
            b"owned-payload",
        )

    assert captured.value is primary
    assert captured_temporary is not None
    assert not captured_temporary.exists()
    assert not (tmp_path / "write-failure.bin").exists()


def test_checkpoint_identity_cleanup_preserves_replaced_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "latest-owned.tmp"
    displaced = tmp_path / "latest-displaced-owned.tmp"
    owned = b"owned-latest-temporary"
    foreign = b"foreign-latest-temporary"
    path.write_bytes(owned)
    expected_identity = checkpoint_module._regular_path_identity(path)
    real_unlink = checkpoint_module.durable_unlink
    injected = False

    def replace_before_unlink(
        candidate: Path,
        *,
        expected_identity: tuple[int, int, int, int, int, int] | None = None,
    ) -> None:
        nonlocal injected
        injected = True
        os.replace(candidate, displaced)
        Path(candidate).write_bytes(foreign)
        if expected_identity is None:
            real_unlink(candidate)
        else:
            real_unlink(candidate, expected_identity=expected_identity)

    monkeypatch.setattr(checkpoint_module, "durable_unlink", replace_before_unlink)

    checkpoint_module._remove_file_if_identity(path, expected_identity)

    assert injected is True
    assert path.read_bytes() == foreign
    assert displaced.read_bytes() == owned


def test_checkpoint_pending_cleanup_preserves_replaced_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = tmp_path / (".pending-update-00000001-" + "a" * 32)
    pending.mkdir()
    member = pending / "manifest.json"
    displaced = tmp_path / "displaced-owned-manifest.json"
    owned = b"owned-pending-member"
    foreign = b"foreign-pending-member"
    member.write_bytes(owned)
    directory_identity = checkpoint_module._directory_path_identity(pending)
    member_identity = checkpoint_module._regular_path_identity(member)
    real_unlink = checkpoint_module.durable_unlink
    injected = False

    def replace_before_unlink(
        candidate: Path,
        *,
        expected_identity: tuple[int, int, int, int, int, int],
    ) -> None:
        nonlocal injected
        injected = True
        os.replace(candidate, displaced)
        Path(candidate).write_bytes(foreign)
        real_unlink(candidate, expected_identity=expected_identity)

    monkeypatch.setattr(checkpoint_module, "durable_unlink", replace_before_unlink)

    checkpoint_module._remove_known_pending_files(
        pending,
        directory_identity=directory_identity,
        member_identities={member.name: member_identity},
    )

    assert injected is True
    assert member.read_bytes() == foreign
    assert displaced.read_bytes() == owned
    assert pending.is_dir()


@pytest.mark.parametrize("close_fails", (False, True))
def test_checkpoint_member_write_failure_cleans_bound_leaf_without_masking_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    close_fails: bool,
) -> None:
    primary = OSError("injected checkpoint member write failure")
    secondary = OSError("injected checkpoint member close failure")
    target_descriptor: int | None = None
    real_close = checkpoint_module.os.close

    def fail_write(descriptor: int, _payload: object) -> int:
        nonlocal target_descriptor
        target_descriptor = descriptor
        raise primary

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        if close_fails and descriptor == target_descriptor:
            raise secondary

    monkeypatch.setattr(checkpoint_module.os, "write", fail_write)
    monkeypatch.setattr(checkpoint_module.os, "close", close_then_fail)
    policy, optimizer = _policy_and_optimizer(seed=29)
    manager = _stage6_manager(tmp_path)

    with pytest.raises(CheckpointError) as captured:
        manager.prepare_complete(
            **_stage4_checkpoint_kwargs(
                policy=policy,
                optimizer=optimizer,
                update_step=1,
                config_sha256=STAGE6_CONFIG_SHA,
                lineage=STAGE6_LINEAGE,
            ),
            safety_contract=STAGE6_SAFETY_CONTRACT,
        )

    assert captured.value.__cause__ is primary
    assert not tuple(manager.root.glob(".pending-update-00000001-*"))


def test_checkpoint_pending_rmdir_preserves_replaced_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = tmp_path / (".pending-update-00000001-" + "b" * 32)
    displaced = tmp_path / "displaced-owned-pending"
    pending.mkdir()
    directory_identity = checkpoint_module._directory_path_identity(pending)
    foreign_marker = pending / "foreign-marker.bin"
    real_rmdir = checkpoint_module.durable_rmdir
    injected = False

    def replace_before_rmdir(
        candidate: Path,
        *,
        expected_identity: tuple[int, int, int, int],
    ) -> None:
        nonlocal injected
        injected = True
        os.replace(candidate, displaced)
        Path(candidate).mkdir()
        foreign_marker.write_bytes(b"foreign-directory-bytes")
        real_rmdir(candidate, expected_identity=expected_identity)

    monkeypatch.setattr(checkpoint_module, "durable_rmdir", replace_before_rmdir)

    checkpoint_module._remove_known_pending_files(
        pending,
        directory_identity=directory_identity,
        member_identities={},
    )

    assert injected is True
    assert foreign_marker.read_bytes() == b"foreign-directory-bytes"
    assert displaced.is_dir()


def test_checkpoint_namespace_durability_order_covers_pending_publish_and_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str, str | None]] = []
    real_write = checkpoint_module._write_new_fsynced

    def record_write(path: Path, payload: bytes):
        identity = real_write(path, payload)
        events.append(("write", path.name, None))
        return identity

    def durable_mkdir(path: Path) -> None:
        events.append(("mkdir", path.name, None))
        path.mkdir(parents=False, exist_ok=False)

    def durable_fsync_directory(path: Path) -> None:
        events.append(("fsync-directory", path.name, None))

    def durable_rename(
        source: Path,
        destination: Path,
        *,
        expected_source_identity: tuple[int, int, int, int],
        replace_existing: bool = False,
    ) -> None:
        events.append(("rename", source.name, destination.name))
        assert checkpoint_module._directory_path_identity(source) == (
            expected_source_identity
        )
        assert replace_existing is False
        os.replace(source, destination)

    def durable_replace(
        source: Path,
        destination: Path,
        *,
        expected_source_identity: tuple[int, int, int, int, int, int],
        replace_existing: bool = True,
    ) -> None:
        events.append(("replace", source.name, destination.name))
        assert checkpoint_module._regular_path_identity(source) == (
            expected_source_identity
        )
        assert replace_existing is True
        os.replace(source, destination)

    monkeypatch.setattr(checkpoint_module, "_write_new_fsynced", record_write)
    monkeypatch.setattr(
        checkpoint_module,
        "durable_mkdir",
        durable_mkdir,
        raising=False,
    )
    monkeypatch.setattr(
        checkpoint_module,
        "durable_fsync_directory",
        durable_fsync_directory,
        raising=False,
    )
    monkeypatch.setattr(
        checkpoint_module,
        "durable_rename",
        durable_rename,
        raising=False,
    )
    monkeypatch.setattr(
        checkpoint_module,
        "durable_replace",
        durable_replace,
        raising=False,
    )
    policy, optimizer = _policy_and_optimizer(seed=24)
    manager = _stage6_manager(tmp_path)

    prepared = manager.prepare_complete(
        **_stage4_checkpoint_kwargs(
            policy=policy,
            optimizer=optimizer,
            update_step=1,
            config_sha256=STAGE6_CONFIG_SHA,
            lineage=STAGE6_LINEAGE,
        ),
        safety_contract=STAGE6_SAFETY_CONTRACT,
    )
    manager.publish_prepared(prepared)

    labels = [event[0] for event in events]
    member_write_indexes = [
        index
        for index, event in enumerate(events)
        if event[0] == "write" and event[1] in {
            "checkpoint.pt",
            "manifest.json",
            "complete.json",
        }
    ]
    pending_flush_indexes = [
        index
        for index, event in enumerate(events)
        if event[:2] == ("fsync-directory", prepared.pending_directory.name)
    ]
    assert pending_flush_indexes, events
    pending_flush_index = pending_flush_indexes[0]
    rename_index = labels.index("rename")
    latest_replace_index = next(
        index
        for index, event in enumerate(events)
        if event[0] == "replace" and event[2] == "latest.json"
    )
    assert labels[0] == "mkdir"
    assert len(member_write_indexes) == 3
    assert max(member_write_indexes) < pending_flush_index < rename_index
    assert rename_index < latest_replace_index


def test_checkpoint_pending_directory_flush_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = OSError("injected pending directory flush failure")

    def fail_pending_flush(path: Path) -> None:
        if path.name.startswith(".pending-update-"):
            raise failure

    monkeypatch.setattr(
        checkpoint_module,
        "durable_fsync_directory",
        fail_pending_flush,
        raising=False,
    )
    policy, optimizer = _policy_and_optimizer(seed=25)
    manager = _stage6_manager(tmp_path)

    with pytest.raises(CheckpointError, match="durab|directory|flush"):
        manager.prepare_complete(
            **_stage4_checkpoint_kwargs(
                policy=policy,
                optimizer=optimizer,
                update_step=1,
                config_sha256=STAGE6_CONFIG_SHA,
                lineage=STAGE6_LINEAGE,
            ),
            safety_contract=STAGE6_SAFETY_CONTRACT,
        )

    assert not (manager.root / "update-00000001").exists()


@pytest.mark.parametrize("primitive_name", ("durable_rename", "durable_replace"))
def test_checkpoint_publish_and_latest_fail_closed_on_namespace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primitive_name: str,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=26)
    manager = _stage6_manager(tmp_path)
    prepared = manager.prepare_complete(
        **_stage4_checkpoint_kwargs(
            policy=policy,
            optimizer=optimizer,
            update_step=1,
            config_sha256=STAGE6_CONFIG_SHA,
            lineage=STAGE6_LINEAGE,
        ),
        safety_contract=STAGE6_SAFETY_CONTRACT,
    )
    failure = OSError(f"injected {primitive_name} failure")

    def fail_closed(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(
        checkpoint_module,
        primitive_name,
        fail_closed,
        raising=False,
    )

    with pytest.raises(CheckpointError, match="publish|latest|durab|namespace"):
        manager.publish_prepared(prepared)

    if primitive_name == "durable_rename":
        assert prepared.pending_directory.is_dir()
        assert not prepared.destination.exists()
    else:
        assert prepared.destination.is_dir()
        assert not (manager.root / "latest.json").exists()


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


def test_failed_optimizer_restore_leaves_policy_optimizer_and_rng_unchanged(
    tmp_path: Path,
) -> None:
    source_policy, source_optimizer = _policy_and_optimizer(seed=31)
    manager = _manager(tmp_path)
    _save(
        manager,
        policy=source_policy,
        optimizer=source_optimizer,
        update_step=1,
    )

    torch.manual_seed(37)
    target_policy = _CheckpointPolicy()
    target_optimizer = torch.optim.AdamW(
        (
            {"params": tuple(target_policy.linear.parameters())},
            {"params": (target_policy.theta,)},
        ),
        lr=7.0e-4,
    )
    loss = target_policy.linear(torch.ones((2, 3))).square().mean()
    loss.backward()
    target_optimizer.step()
    target_optimizer.zero_grad(set_to_none=True)

    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    policy_hash_before = policy_state_sha256(target_policy)
    optimizer_before = copy.deepcopy(target_optimizer.state_dict())
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = tuple(
        state.clone() for state in torch.cuda.get_rng_state_all()
    ) if torch.cuda.is_available() else ()

    with pytest.raises(ValueError):
        manager.load_complete(
            update_step=1,
            policy=target_policy,
            optimizer=target_optimizer,
            expected_config_sha256=CONFIG_SHA,
            expected_lineage=LINEAGE,
        )

    assert policy_state_sha256(target_policy) == policy_hash_before
    optimizer_after = target_optimizer.state_dict()
    assert optimizer_after["param_groups"] == optimizer_before["param_groups"]
    assert set(optimizer_after["state"]) == set(optimizer_before["state"])
    for parameter_id, expected_fields in optimizer_before["state"].items():
        actual_fields = optimizer_after["state"][parameter_id]
        assert set(actual_fields) == set(expected_fields)
        for name, expected_value in expected_fields.items():
            actual_value = actual_fields[name]
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value
    assert random.getstate() == python_rng_before
    numpy_rng_after = np.random.get_state()
    assert numpy_rng_after[0] == numpy_rng_before[0]
    assert np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
    assert numpy_rng_after[2:] == numpy_rng_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    if cuda_rng_before:
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                torch.cuda.get_rng_state_all(),
                cuda_rng_before,
                strict=True,
            )
        )


def test_prepare_is_hidden_until_publish_and_replaces_stale_pending(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer()
    manager = _manager(tmp_path)
    _save(manager, policy=policy, optimizer=optimizer, update_step=1)
    latest_path = manager.root / "latest.json"
    latest_before = latest_path.read_bytes()
    arguments = _stage4_checkpoint_kwargs(
        policy=policy,
        optimizer=optimizer,
        update_step=2,
    )

    stale = manager.prepare_complete(**arguments)

    canonical = manager.root / "update-00000002"
    assert not canonical.exists()
    assert latest_path.read_bytes() == latest_before
    assert stale.pending_directory.name.startswith(".pending-update-00000002-")
    assert {path.name for path in stale.pending_directory.iterdir()} == {
        "checkpoint.pt",
        "manifest.json",
        "complete.json",
    }
    assert stale.complete_marker_sha256 == hashlib.sha256(
        stale.pending_complete_marker_path.read_bytes()
    ).hexdigest()

    restarted = _manager(tmp_path)
    prepared = restarted.prepare_complete(**arguments)

    assert not stale.pending_directory.exists()
    assert prepared.pending_directory != stale.pending_directory
    assert not canonical.exists()
    assert latest_path.read_bytes() == latest_before

    receipt = restarted.publish_prepared(prepared)

    assert receipt.directory == canonical
    assert receipt.complete_marker_path.is_file()
    assert json.loads(latest_path.read_text(encoding="utf-8"))["update_step"] == 2
    with pytest.raises(CheckpointError, match="prepared|pending|identity"):
        restarted.publish_prepared(stale)


def test_recover_accepted_checkpoint_publishes_exact_hidden_pending(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=71)
    manager = _stage6_manager(tmp_path)
    prepared = manager.prepare_complete(
        policy=policy,
        optimizer=optimizer,
        update_step=1,
        normalization_stats={},
        scenario_sampler_state={"workers": list(range(8))},
        vector_env_states=[{"worker": index} for index in range(8)],
        best_record={},
        versions=VERSIONS,
        top_m_config={"top_m": 1024},
        scale_profile="Standard v1",
        training_config={"updates_per_seed": 100},
        config_sha256=STAGE6_CONFIG_SHA,
        lineage=STAGE6_LINEAGE,
        eval_metrics={},
        safety_contract=STAGE6_SAFETY_CONTRACT,
    )
    canonical = manager.root / "update-00000001"
    assert not canonical.exists()
    assert not (manager.root / "latest.json").exists()

    restarted = _stage6_manager(tmp_path)
    with pytest.raises(CheckpointError, match="identity mismatch"):
        restarted.recover_accepted_checkpoint(
            update_step=1,
            checkpoint_sha256="f" * 64,
            complete_marker_sha256=prepared.complete_marker_sha256,
            policy_state_sha256=prepared.policy_state_sha256,
        )
    assert prepared.pending_directory.is_dir()
    assert not canonical.exists()
    assert not (manager.root / "latest.json").exists()

    receipt = restarted.recover_accepted_checkpoint(
        update_step=1,
        checkpoint_sha256=prepared.checkpoint_sha256,
        complete_marker_sha256=prepared.complete_marker_sha256,
        policy_state_sha256=prepared.policy_state_sha256,
    )

    assert receipt.directory == canonical
    assert receipt.checkpoint_sha256 == prepared.checkpoint_sha256
    assert receipt.policy_state_sha256 == prepared.policy_state_sha256
    assert hashlib.sha256(receipt.complete_marker_path.read_bytes()).hexdigest() == (
        prepared.complete_marker_sha256
    )
    assert not prepared.pending_directory.exists()
    assert json.loads(
        (manager.root / "latest.json").read_text(encoding="utf-8")
    )["update_step"] == 1


@pytest.mark.parametrize(
    ("fault_phase", "with_previous_latest"),
    (
        ("after_publish_rename", False),
        ("before_latest_replace", True),
    ),
)
def test_recover_accepted_checkpoint_repairs_publish_fault_from_exact_identity(
    tmp_path: Path,
    fault_phase: str,
    with_previous_latest: bool,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=72)
    stable = _stage6_manager(tmp_path)
    update_step = 2 if with_previous_latest else 1
    if with_previous_latest:
        _save_stage6(
            stable,
            policy=policy,
            optimizer=optimizer,
            update_step=1,
        )
    prepared = stable.prepare_complete(
        policy=policy,
        optimizer=optimizer,
        update_step=update_step,
        normalization_stats={},
        scenario_sampler_state={"workers": list(range(8))},
        vector_env_states=[{"worker": index} for index in range(8)],
        best_record={},
        versions=VERSIONS,
        top_m_config={"top_m": 1024},
        scale_profile="Standard v1",
        training_config={"updates_per_seed": 100},
        config_sha256=STAGE6_CONFIG_SHA,
        lineage=STAGE6_LINEAGE,
        eval_metrics={},
        safety_contract=STAGE6_SAFETY_CONTRACT,
    )

    def inject(phase: str) -> None:
        if phase == fault_phase:
            raise OSError(f"injected accepted publish failure: {phase}")

    failing = _stage6_manager(tmp_path, fault_injector=inject)
    with pytest.raises((OSError, CheckpointError)):
        failing.recover_accepted_checkpoint(
            update_step=update_step,
            checkpoint_sha256=prepared.checkpoint_sha256,
            complete_marker_sha256=prepared.complete_marker_sha256,
            policy_state_sha256=prepared.policy_state_sha256,
        )
    canonical = stable.root / f"update-{update_step:08d}"
    assert canonical.is_dir()
    latest_before = (
        json.loads((stable.root / "latest.json").read_text(encoding="utf-8"))
        if (stable.root / "latest.json").is_file()
        else None
    )
    assert latest_before is None or latest_before["update_step"] < update_step

    receipt = _stage6_manager(tmp_path).recover_accepted_checkpoint(
        update_step=update_step,
        checkpoint_sha256=prepared.checkpoint_sha256,
        complete_marker_sha256=prepared.complete_marker_sha256,
        policy_state_sha256=prepared.policy_state_sha256,
    )

    assert receipt.directory == canonical
    assert receipt.checkpoint_sha256 == prepared.checkpoint_sha256
    assert json.loads(
        (stable.root / "latest.json").read_text(encoding="utf-8")
    )["update_step"] == update_step


def test_accepted_publish_fault_before_rename_preserves_pending_for_recovery(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=74)
    stable = _stage6_manager(tmp_path)
    prepared = stable.prepare_complete(
        policy=policy,
        optimizer=optimizer,
        update_step=1,
        normalization_stats={},
        scenario_sampler_state={"workers": list(range(8))},
        vector_env_states=[{"worker": index} for index in range(8)],
        best_record={},
        versions=VERSIONS,
        top_m_config={"top_m": 1024},
        scale_profile="Standard v1",
        training_config={"updates_per_seed": 100},
        config_sha256=STAGE6_CONFIG_SHA,
        lineage=STAGE6_LINEAGE,
        eval_metrics={},
        safety_contract=STAGE6_SAFETY_CONTRACT,
    )

    def inject(phase: str) -> None:
        if phase == "before_publish_rename":
            raise OSError("injected accepted crash before rename")

    with pytest.raises(OSError, match="before rename"):
        _stage6_manager(tmp_path, fault_injector=inject).recover_accepted_checkpoint(
            update_step=1,
            checkpoint_sha256=prepared.checkpoint_sha256,
            complete_marker_sha256=prepared.complete_marker_sha256,
            policy_state_sha256=prepared.policy_state_sha256,
        )
    assert prepared.pending_directory.is_dir()
    assert not (stable.root / "update-00000001").exists()

    recovered = _stage6_manager(tmp_path).recover_accepted_checkpoint(
        update_step=1,
        checkpoint_sha256=prepared.checkpoint_sha256,
        complete_marker_sha256=prepared.complete_marker_sha256,
        policy_state_sha256=prepared.policy_state_sha256,
    )
    assert recovered.directory == stable.root / "update-00000001"


def test_recover_accepted_checkpoint_is_idempotent_and_rejects_latest_drift(
    tmp_path: Path,
) -> None:
    policy, optimizer = _policy_and_optimizer(seed=73)
    manager = _stage6_manager(tmp_path)
    prepared = manager.prepare_complete(
        policy=policy,
        optimizer=optimizer,
        update_step=1,
        normalization_stats={},
        scenario_sampler_state={"workers": list(range(8))},
        vector_env_states=[{"worker": index} for index in range(8)],
        best_record={},
        versions=VERSIONS,
        top_m_config={"top_m": 1024},
        scale_profile="Standard v1",
        training_config={"updates_per_seed": 100},
        config_sha256=STAGE6_CONFIG_SHA,
        lineage=STAGE6_LINEAGE,
        eval_metrics={},
        safety_contract=STAGE6_SAFETY_CONTRACT,
    )
    published = manager.publish_prepared(prepared)

    recovered = _stage6_manager(tmp_path).recover_accepted_checkpoint(
        update_step=1,
        checkpoint_sha256=prepared.checkpoint_sha256,
        complete_marker_sha256=prepared.complete_marker_sha256,
        policy_state_sha256=prepared.policy_state_sha256,
    )
    assert recovered == published

    latest_path = manager.root / "latest.json"
    drifted = {
        "schema_version": "stage6_checkpoint_latest/v1",
        "update_step": 1,
        "directory": "update-00000001",
        "manifest_sha256": published.manifest_sha256,
        "checkpoint_sha256": "f" * 64,
    }
    drifted_bytes = ArtifactStore.canonical_json_bytes(drifted)
    latest_path.write_bytes(drifted_bytes)
    with pytest.raises(CheckpointError, match="latest.*drift|drift.*latest"):
        _stage6_manager(tmp_path).recover_accepted_checkpoint(
            update_step=1,
            checkpoint_sha256=prepared.checkpoint_sha256,
            complete_marker_sha256=prepared.complete_marker_sha256,
            policy_state_sha256=prepared.policy_state_sha256,
        )
    assert latest_path.read_bytes() == drifted_bytes


@pytest.mark.parametrize(
    "fault_phase",
    (
        "after_payload_write",
        "after_manifest_write",
        "after_pending_complete_write",
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
