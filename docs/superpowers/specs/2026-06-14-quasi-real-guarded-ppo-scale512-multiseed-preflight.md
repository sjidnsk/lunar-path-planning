# Quasi-Real Guarded PPO Scale-512 Multi-Seed Preflight v1

## Summary

This stage follows the passed `Quasi-Real Guarded PPO Horizon-5 Batch Expansion
v1` evidence and adds the formal PPO preflight gate. It is not formal PPO and
does not publish or replace a checkpoint. It verifies whether the quasi-real
guarded rollout evidence has enough real diversity for a later training run:
`horizon>=5`, at least 512 PPO-trainable transitions, at least 512 unique
trainable contexts, and three tiny seed-level PPO smoke checks.

## Artifacts

- `configs/quasi_real_guarded_ppo_scale512_multiseed_preflight_v1.json`
- `scripts/run_quasi_real_guarded_ppo_scale512_multiseed_preflight.py`
- `scripts/run_quasi_real_guarded_ppo_scale512_multiseed_preflight.sh`
- `scripts/run_quasi_real_guarded_ppo_scale512_multiseed_preflight_closure.sh`
- `tests/test_quasi_real_guarded_ppo_scale512_multiseed_preflight.py`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_scale512_multiseed_preflight_v1/quasi-real-guarded-ppo-scale512-multiseed-preflight-summary.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_scale512_multiseed_preflight_v1/scale512-trainable-capacity-report.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_scale512_multiseed_preflight_v1/scale512-trainable-contexts.jsonl`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_scale512_multiseed_preflight_v1/scale512-seed-summaries.jsonl`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_scale512_multiseed_preflight_v1/scale512-readiness-validate-only.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_scale512_multiseed_preflight_v1/scale512-multiseed-preflight-report.md`

## Contract

The runner reads the Horizon-5 summary and expanded step ledger. Horizon-5 must
be passed, must have `horizon>=5`, replay stability, teacher agreement at least
0.95, and zero controlled regression.

Only train split, controlled-policy, gate-clean, fully materialized, finite
steps can count as PPO-trainable. Validation/test split, source fallback,
teacher fallback, raw probe rejection, non-empty gate reasons, and
path-planner/IRIS/GCS diagnostics remain diagnostic-only.

The preflight does not duplicate contexts to reach scale. If fewer than 512
unique trainable contexts are available, it fails with
`insufficient_quasi_real_trainable_capacity` and skips seed smoke.

When capacity is sufficient, seeds `[0,1,2]` must each pass tiny full-batch PPO
smoke with old log_prob/value reconstruction error `<=1e-4`, finite
loss/grad/reward/return/advantage, `abs(approx_kl)<=0.25`,
`max_grad_norm_after_clip<=1.0`, teacher agreement at least 0.95, zero
controlled regression, and post-update guarded collector trainable count at
least 512.

## Current Evidence

The current Horizon-5 evidence is passed but too small for Scale-512:

- `horizon=5`
- `ppo_trainable_transition_count=162`
- unique quasi-real trainable contexts: `36`

Therefore the correct current Scale-512 closure result is a failed preflight
with `insufficient_quasi_real_trainable_capacity`. This is a capacity blocker,
not PPO optimizer instability and not a teacher-skill regression.

## Readiness

`run_policy_training_readiness_review.py` accepts:

```bash
--quasi-real-guarded-ppo-scale512-multiseed-preflight-summary
```

Only a passed Scale-512 summary advances readiness to
`quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated`. A failed
capacity gate keeps readiness blocked and points to upstream quasi-real
trainable context expansion.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_guarded_ppo_scale512_multiseed_preflight.py \
  tests/test_quasi_real_guarded_ppo_horizon5_batch_expansion.py \
  tests/test_quasi_real_guarded_ppo_stability_replay.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_guarded_ppo_scale512_multiseed_preflight_closure.sh

jq '{status, reason_codes, horizon, ppo_trainable_transition_count, unique_trainable_context_count, seed_count, passed_seed_count, readiness_status, controlled_regression_count}' \
  outputs/path_feedback_batch_quasi_real_guarded_ppo_scale512_multiseed_preflight_v1/quasi-real-guarded-ppo-scale512-multiseed-preflight-summary.json

git diff --check
```

## Non-Goals

No formal PPO rollout, checkpoint publication, default-policy replacement,
network/action-space/default-A* change, gate relaxation, Ackermann-feasible
trajectory claim, policy performance claim, formal-training-ready claim, or
IRIS/GCS/path-planner diagnostic training release is made by this stage.
