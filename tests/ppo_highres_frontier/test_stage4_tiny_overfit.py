from __future__ import annotations

import inspect

import torch

from lunar_exploration_ppo.ppo.tiny_overfit import (
    TINY_OVERFIT_SEEDS,
    run_tiny_overfit_acceptance,
)


def test_tiny_overfit_four_contexts_three_seeds_passes_real_ppo_path() -> None:
    assert torch.cuda.is_available()
    result = run_tiny_overfit_acceptance(device="cuda")

    assert result.context_count == 4
    assert tuple(seed.seed for seed in result.seeds) == TINY_OVERFIT_SEEDS
    assert len(result.seeds) == 3
    assert result.passed
    for seed in result.seeds:
        assert seed.passed
        assert seed.stop_update <= 200
        assert seed.correct_frontier_probability_min >= 0.95
        assert seed.theta_error_rad_max <= 0.1
        assert seed.value_mse_reduction >= 0.90
        assert seed.ppo_update_count == seed.stop_update
        assert seed.rollout_batch_count == seed.stop_update
        assert len(seed.curve) == seed.stop_update
        assert all(point.trainable_transition_count == 1024 for point in seed.curve)
        assert all(point.initial_ratio_max_abs_error <= 1.0e-5 for point in seed.curve)
        assert all(point.post_clip_grad_norm <= 0.500001 for point in seed.curve)


def test_tiny_overfit_source_does_not_use_supervised_optimizer_or_weight_writes() -> None:
    import lunar_exploration_ppo.ppo.tiny_overfit as tiny

    source = inspect.getsource(tiny)
    assert "PPOTrainer(" in source
    assert "RolloutBuffer(" in source
    assert ".update(batch)" in source
    assert "CrossEntropyLoss" not in source
    assert "MSELoss" not in source
    assert ".data" not in source
    assert "copy_(" not in source
