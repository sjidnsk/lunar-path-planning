"""Stage 4 frozen config, inherited authority, and machine workflow contracts."""

from __future__ import annotations

import copy
import hashlib
import inspect
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from lunar_exploration_ppo.configs.stage4 import Stage4Config, load_stage4_config
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.workflows.stage1_artifacts import FrozenFileSnapshot
from lunar_exploration_ppo.workflows.stage4 import (
    ABSENT_AUTHORITY_SHA256,
    STAGE3_APPROVAL_SHA256,
    STAGE3_GATE_SHA256,
    STAGE3_REVIEW_SHA256,
    FrozenStage3AuthorityHandle,
    Stage4WorkflowError,
    _validate_stage3_gate_history,
    verify_frozen_stage3_authority,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/ppo_highres_frontier_stage4_v1.json"
STAGE3_GATE = Path(
    "D:/xunce/out/ppo_frontier/"
    "s3-task4-cifix-final-20260713T213739Z/s3/gate.json"
)


def test_stage4_config_freezes_rollout_ppo_checkpoint_and_acceptance() -> None:
    config = load_stage4_config(CONFIG)
    assert config.stage_id == "ppo_highres_frontier_stage4_rollout_ppo_update/v1"
    assert config.stage3_authority.commit_sha256 == (
        "4df7be92cd6517e77c648e890bf7d32c9fcd560b"
    )
    assert config.stage3_authority.gate_sha256 == STAGE3_GATE_SHA256
    assert config.stage3_authority.approval_sha256 == STAGE3_APPROVAL_SHA256
    assert config.stage3_authority.review_sha256 == STAGE3_REVIEW_SHA256
    assert config.rollout.env_count == 8
    assert config.rollout.trainable_steps_per_env == 128
    assert config.rollout.batch_size == 1024
    assert config.rollout.gamma == 0.995
    assert config.rollout.gae_lambda == 0.95
    assert config.ppo.model_dump() == {
        "ppo_epochs": 4,
        "effective_minibatch_size": 256,
        "physical_microbatch_size": 32,
        "shuffle": True,
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "adam_eps": 0.00001,
        "weight_decay": 0.0001,
        "clip_eps": 0.2,
        "value_clip_eps": 0.2,
        "value_loss_coef": 0.5,
        "frontier_entropy_coef": 0.01,
        "theta_entropy_enabled": False,
        "max_grad_norm": 0.5,
        "target_kl": 0.03,
        "compute_dtype": "float32",
        "amp_enabled": False,
    }
    assert config.acceptance.smoke_updates == 3
    assert config.acceptance.tiny_overfit_contexts == 4
    assert config.acceptance.tiny_overfit_seeds == (17, 29, 43)
    assert config.acceptance.tiny_overfit_max_updates == 200
    assert config.device == "cuda"
    assert config.allow_cpu_fallback is False
    assert config.output_root == "D:/xunce/out/ppo_frontier"

    payload = config.model_dump(mode="json")
    payload["ppo"]["ppo_epochs"] = 3
    with pytest.raises(ValidationError, match="contract drifted"):
        Stage4Config.model_validate(payload)


def test_stage3_gate_history_uses_exact_five_state_hash_chain() -> None:
    gate = ArtifactStore.canonical_json_bytes(
        __import__("json").loads(STAGE3_GATE.read_text(encoding="utf-8"))
    )
    parsed = __import__("json").loads(gate)
    _validate_stage3_gate_history(parsed["history"], parsed["bindings"])

    tampered = copy.deepcopy(parsed["history"])
    tampered[2]["bindings"]["review_sha256"] = ABSENT_AUTHORITY_SHA256
    with pytest.raises(Stage4WorkflowError, match="history"):
        _validate_stage3_gate_history(tampered, parsed["bindings"])


def test_frozen_stage3_authority_handle_detects_content_and_aba_replacement(
    tmp_path: Path,
) -> None:
    authority = tmp_path / "authority.json"
    authority.write_bytes(b"authority-a\n")
    snapshot = FrozenFileSnapshot.capture(authority)
    handle = FrozenStage3AuthorityHandle(
        identity={"verified": True},
        snapshots=(("authority", snapshot),),
        stage2_authority=None,
    )

    authority.write_bytes(b"authority-b\n")
    with pytest.raises(Stage4WorkflowError, match="changed"):
        handle.require_current()

    authority.write_bytes(b"authority-a\n")
    with pytest.raises(Stage4WorkflowError, match="changed"):
        handle.require_current()


def test_stage4_production_surface_has_no_authority_issuer_or_force() -> None:
    import lunar_exploration_ppo.workflows.stage4 as workflow

    source = inspect.getsource(workflow).lower()
    forbidden = (
        "issue_approval",
        "record_approval",
        "create_gate",
        "write_gate",
        "record_review",
        "--force",
    )
    assert not any(token in source for token in forbidden)
    assert "review.json" not in workflow.STAGE4_ROOT_ARTIFACTS
    assert "approval.json" not in workflow.STAGE4_ROOT_ARTIFACTS
    assert "gate.json" not in workflow.STAGE4_ROOT_ARTIFACTS
