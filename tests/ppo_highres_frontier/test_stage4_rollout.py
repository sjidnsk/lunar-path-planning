from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from lunar_exploration_ppo.policy.observation import PolicyObservation
from lunar_exploration_ppo.utils.geometry import CellXY


def test_stage4_rollout_module_is_available() -> None:
    assert importlib.util.find_spec("lunar_exploration_ppo.ppo.rollout") is not None


def test_stage4_fixed_public_interfaces_are_exported() -> None:
    from lunar_exploration_ppo.ppo import (
        CheckpointManager,
        PPOTrainer,
        RolloutBatch,
        RolloutBuffer,
    )

    assert callable(RolloutBuffer.finalize)
    assert callable(PPOTrainer.update)
    assert callable(CheckpointManager.save_complete)
    assert callable(CheckpointManager.load_last_complete)
    assert RolloutBatch.__name__ == "RolloutBatch"


def _observation() -> tuple[PolicyObservation, dict[str, np.ndarray]]:
    arrays = {
        "prior_channels": np.arange(7 * 2 * 2, dtype=np.float32).reshape(7, 2, 2),
        "coverage_summary": np.arange(8 * 2 * 2, dtype=np.float32).reshape(8, 2, 2),
        "local_crop": np.arange(8 * 3 * 3, dtype=np.float32).reshape(8, 3, 3),
        "frontier_features": np.arange(3 * 22, dtype=np.float32).reshape(3, 22),
        "pose_features": np.arange(6, dtype=np.float32),
        "candidate_mask": np.asarray([True, True, False], dtype=bool),
    }
    return PolicyObservation(**arrays), arrays


def test_candidate_snapshot_deep_copy_round_trip_hash_and_excludes_truth() -> None:
    from lunar_exploration_ppo.ppo.rollout import CandidateSnapshot

    observation, source = _observation()
    snapshot = CandidateSnapshot.capture(
        observation=observation,
        frontier_cells=(CellXY(4, 5), CellXY(6, 7)),
        top_m_selection_summary={
            "top_m_selection_policy": "score_first_top_m/v1",
            "candidate_count_before_top_m": 2,
            "candidate_count_after_top_m": 2,
        },
    )
    payload = snapshot.to_bytes()
    restored = CandidateSnapshot.from_bytes(
        payload,
        expected_sha256=snapshot.sha256,
    )

    source["frontier_features"].setflags(write=True)
    source["frontier_features"][0, 0] = -1234.0

    assert snapshot.frontier_features[0, 0] == 0.0
    assert restored.sha256 == snapshot.sha256
    assert restored.frontier_cells == (CellXY(4, 5), CellXY(6, 7))
    assert restored.top_m_selection_summary == snapshot.top_m_selection_summary
    assert all(not array.flags.writeable for array in restored.array_fields())
    assert np.array_equal(
        restored.to_policy_observation().candidate_mask,
        np.asarray([True, True, False], dtype=bool),
    )
    assert b"coverable_mask" not in payload
    assert b'"truth"' not in payload


def test_candidate_snapshot_rejects_forbidden_truth_and_tampered_bytes() -> None:
    from lunar_exploration_ppo.ppo.rollout import CandidateSnapshot, RolloutContractError

    observation, _ = _observation()
    with pytest.raises(RolloutContractError, match="forbidden"):
        CandidateSnapshot.capture(
            observation=observation,
            frontier_cells=(CellXY(4, 5), CellXY(6, 7)),
            top_m_selection_summary={"truth": {"hidden": True}},
        )

    snapshot = CandidateSnapshot.capture(
        observation=observation,
        frontier_cells=(CellXY(4, 5), CellXY(6, 7)),
        top_m_selection_summary={"candidate_count_after_top_m": 2},
    )
    tampered = bytearray(snapshot.to_bytes())
    tampered[-1] ^= 1
    with pytest.raises(RolloutContractError, match="hash|payload"):
        CandidateSnapshot.from_bytes(
            bytes(tampered),
            expected_sha256=snapshot.sha256,
        )


def _snapshot():
    from lunar_exploration_ppo.ppo.rollout import CandidateSnapshot

    observation, _ = _observation()
    return CandidateSnapshot.capture(
        observation=observation,
        frontier_cells=(CellXY(4, 5), CellXY(6, 7)),
        top_m_selection_summary={
            "top_m_selection_policy": "score_first_top_m/v1",
            "candidate_count_before_top_m": 2,
            "candidate_count_after_top_m": 2,
        },
    )


def _transition(**overrides):
    from lunar_exploration_ppo.ppo.rollout import RolloutTransition

    values = {
        "snapshot": _snapshot(),
        "policy_state_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "lineage": {
            "stage3_commit": "4df7be92cd6517e77c648e890bf7d32c9fcd560b",
            "stage3_gate_sha256": "6ea3dc5bc199dd9370fab4b1a5aac1ebab08d9e42f49c0d01b3aa2b1e9aae0f0",
        },
        "selected_frontier_index": 1,
        "selected_cell_xy": CellXY(6, 7),
        "selected_theta": np.float32(0.25),
        "old_log_prob_frontier": np.float32(-0.75),
        "old_log_prob_theta": np.float32(-1.25),
        "old_log_prob_total": np.float32(-2.0),
        "old_value": np.float32(0.5),
        "reward": np.float32(1.0),
        "done": False,
        "done_reason": "none",
        "planned_path_cells": (CellXY(1, 1), CellXY(2, 2)),
        "path_length_m": np.float32(0.70710677),
        "path_observation_step_m": np.float32(1.0),
        "coverage_gain_cells": 3,
        "coverage_gain_per_meter": np.float32(4.2426405),
        "observation_sample_count": 2,
        "ray_count_per_sample": 91,
        "ray_cell_visit_count": 123,
        "newly_observed_cell_count": 3,
        "planner_validation_summary": {"valid": True, "failure_reason": "none"},
        "execution_observation_summary": {"sample_sources": ["path_tangent", "endpoint_theta"]},
        "frontier_extractor_version": "observed_frontier_segment_landing/v1",
        "observation_schema_version": "policy_observation/v1",
        "action_space_version": "frontier_index_continuous_theta/v1",
        "reward_version": "coverage_gain_terminal_reward/v1",
        "planner_version": "observed_safe_astar/v1",
    }
    values.update(overrides)
    return RolloutTransition.create(**values)


def test_rollout_transition_round_trip_requires_exact_snapshot_action_and_old_policy() -> None:
    from lunar_exploration_ppo.ppo.rollout import RolloutTransition

    transition = _transition()
    restored = RolloutTransition.from_record(
        transition.to_record(),
        snapshot_bytes=transition.snapshot_bytes,
    )

    assert restored.candidate_snapshot_sha256 == _snapshot().sha256
    assert restored.selected_cell_xy == CellXY(6, 7)
    assert restored.snapshot.frontier_features[1, 0] == 22.0
    assert restored.old_log_prob_total == np.float32(-2.0)
    assert restored.policy_state_sha256 == "1" * 64
    assert restored.lineage == transition.lineage


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"selected_frontier_index": 2, "selected_cell_xy": CellXY(9, 9)}, "masked"),
        ({"selected_cell_xy": CellXY(9, 9)}, "selected cell"),
        ({"old_log_prob_total": np.float32(-3.0)}, "factorization"),
        ({"old_value": np.float32(np.nan)}, "finite"),
        ({"policy_state_sha256": "not-a-hash"}, "policy"),
        ({"coverage_gain_per_meter": np.float32(9.0)}, "coverage gain"),
    ],
)
def test_rollout_transition_rejects_invalid_action_logprob_hash_and_nonfinite(
    overrides: dict[str, object],
    message: str,
) -> None:
    from lunar_exploration_ppo.ppo.rollout import RolloutContractError

    with pytest.raises(RolloutContractError, match=message):
        _transition(**overrides)


def test_rollout_transition_record_rejects_missing_or_mismatched_snapshot_fields() -> None:
    from lunar_exploration_ppo.ppo.rollout import RolloutContractError, RolloutTransition

    transition = _transition()
    missing = transition.to_record()
    missing.pop("old_log_prob_theta")
    with pytest.raises(RolloutContractError, match="field set"):
        RolloutTransition.from_record(missing, snapshot_bytes=transition.snapshot_bytes)

    mismatched = transition.to_record()
    mismatched["candidate_snapshot_sha256"] = "3" * 64
    with pytest.raises(RolloutContractError, match="snapshot hash"):
        RolloutTransition.from_record(
            mismatched,
            snapshot_bytes=transition.snapshot_bytes,
        )


def test_gae_matches_hand_calculation_across_terminal_and_bootstrap_boundaries() -> None:
    from lunar_exploration_ppo.ppo.rollout import compute_gae

    rewards = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, -1.0]], dtype=np.float32)
    values = np.asarray([[0.5, 0.2], [0.6, 0.3], [0.7, 0.4]], dtype=np.float32)
    dones = np.asarray([[False, True], [True, False], [False, True]], dtype=bool)
    last_values = np.asarray([0.8, 99.0], dtype=np.float32)

    result = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_values=last_values,
    )

    expected = np.asarray(
        [
            [2.42035, -0.2],
            [1.4, -0.22535],
            [3.096, -1.4],
        ],
        dtype=np.float32,
    )
    assert np.allclose(result.raw_advantages, expected, rtol=0.0, atol=2e-6)
    assert np.allclose(result.returns, expected + values, rtol=0.0, atol=2e-6)
    assert result.normalized_advantages.dtype == np.float32
    assert float(result.normalized_advantages.mean(dtype=np.float32)) == pytest.approx(
        0.0, abs=2e-6
    )
    assert float(result.normalized_advantages.std(dtype=np.float32)) == pytest.approx(
        1.0, abs=2e-6
    )
    assert np.all(np.isfinite(result.returns))


def test_gae_rejects_wrong_dtype_shape_nonfinite_and_degenerate_normalization() -> None:
    from lunar_exploration_ppo.ppo.rollout import RolloutContractError, compute_gae

    valid = {
        "rewards": np.asarray([[1.0], [2.0]], dtype=np.float32),
        "values": np.asarray([[0.0], [0.5]], dtype=np.float32),
        "dones": np.asarray([[False], [True]], dtype=bool),
        "last_values": np.asarray([0.0], dtype=np.float32),
    }
    with pytest.raises(RolloutContractError, match="FP32"):
        compute_gae(**(valid | {"rewards": valid["rewards"].astype(np.float64)}))
    with pytest.raises(RolloutContractError, match="shape"):
        compute_gae(**(valid | {"last_values": np.asarray([0.0, 0.0], dtype=np.float32)}))
    with pytest.raises(RolloutContractError, match="finite"):
        compute_gae(**(valid | {"values": np.asarray([[np.nan], [0.5]], dtype=np.float32)}))
    with pytest.raises(RolloutContractError, match="degenerate"):
        compute_gae(
            rewards=np.zeros((1, 2), dtype=np.float32),
            values=np.zeros((1, 2), dtype=np.float32),
            dones=np.ones((1, 2), dtype=bool),
            last_values=np.zeros((2,), dtype=np.float32),
        )


def test_rollout_buffer_finalizes_exact_time_major_128_by_8_and_clears_storage() -> None:
    from lunar_exploration_ppo.ppo.rollout import RolloutBuffer

    buffer = RolloutBuffer()
    for time_index in range(128):
        for env_index in range(8):
            terminal = time_index == 127
            buffer.add(
                env_index,
                _transition(
                    reward=np.float32(time_index + env_index / 16.0),
                    done=terminal,
                    done_reason="failure_done" if terminal else "none",
                ),
            )

    assert buffer.per_env_counts == (128,) * 8
    assert buffer.transition_count == 1024
    batch = buffer.finalize(np.full((8,), 123.0, dtype=np.float32))

    assert batch.layout_shape == (128, 8)
    assert batch.size == 1024
    assert batch.rewards.shape == (128, 8)
    assert batch.raw_advantages.shape == (128, 8)
    assert batch.normalized_advantages.shape == (128, 8)
    assert batch.returns.shape == (128, 8)
    assert batch.rewards[17, 3] == np.float32(17 + 3 / 16.0)
    assert batch.transitions[17 * 8 + 3].reward == batch.rewards[17, 3]
    assert float(batch.normalized_advantages.mean(dtype=np.float32)) == pytest.approx(
        0.0, abs=2e-5
    )
    assert float(batch.normalized_advantages.std(dtype=np.float32)) == pytest.approx(
        1.0, abs=2e-5
    )
    assert buffer.finalized
    assert buffer.transition_count == 0
    assert buffer.per_env_counts == (0,) * 8


def test_rollout_buffer_rejects_incomplete_overquota_and_mixed_policy_or_config() -> None:
    from lunar_exploration_ppo.ppo.rollout import RolloutBuffer, RolloutContractError

    incomplete = RolloutBuffer()
    incomplete.add(0, _transition())
    with pytest.raises(RolloutContractError, match="quota"):
        incomplete.finalize(np.zeros((8,), dtype=np.float32))

    mixed_policy = RolloutBuffer()
    mixed_policy.add(0, _transition())
    with pytest.raises(RolloutContractError, match="policy"):
        mixed_policy.add(1, _transition(policy_state_sha256="3" * 64))

    mixed_config = RolloutBuffer()
    mixed_config.add(0, _transition())
    with pytest.raises(RolloutContractError, match="config"):
        mixed_config.add(1, _transition(config_sha256="4" * 64))

    overquota = RolloutBuffer()
    for _ in range(128):
        overquota.add(0, _transition())
    with pytest.raises(RolloutContractError, match="quota"):
        overquota.add(0, _transition())


def test_rollout_batch_is_single_use_and_rejects_stale_policy() -> None:
    from lunar_exploration_ppo.ppo.rollout import RolloutBuffer, RolloutContractError

    buffer = RolloutBuffer()
    transition = _transition()
    for _ in range(128):
        for env_index in range(8):
            buffer.add(env_index, transition)
    batch = buffer.finalize(np.zeros((8,), dtype=np.float32))

    with pytest.raises(RolloutContractError, match="stale policy"):
        batch.claim_for_update("9" * 64)
    batch.claim_for_update("1" * 64)
    with pytest.raises(RolloutContractError, match="claimed"):
        batch.claim_for_update("1" * 64)
    batch.mark_consumed()
    assert batch.consumed
    with pytest.raises(RolloutContractError, match="consumed"):
        batch.claim_for_update("1" * 64)
