# Limited PPO Update Smoke v1

## Summary

Current readiness is `ppo_rollout_collector_dry_run_evaluated`: the collector
materializes 36 PPO-trainable policy-controlled transitions from sequential
canary evidence, with invalid/empty action mask, missing log-prob/value,
non-finite reward, source-fallback trainable, path/risk/safety/contract, and
source-selection regression all at 0. This stage performs one tiny local PPO
optimizer smoke from the same experimental checkpoint that collected those
transitions, then reruns post-update gates.

This is not formal PPO rollout and not a model release.

## Artifacts

- `configs/limited_ppo_update_smoke_v1.json`
- `scripts/run_limited_ppo_update_smoke.py`
- `scripts/run_limited_ppo_update_smoke.sh`
- `scripts/run_limited_ppo_update_smoke_closure.sh`
- `outputs/path_feedback_batch_limited_ppo_update_smoke_v1/`

The smoke output root contains:

- `limited-ppo-update-smoke-summary.json`
- `limited-ppo-update-training-curves.json`
- `limited-ppo-update-diagnostics.json`
- `experimental-hybrid-policy-candidate.pt`
- `experimental-hybrid-policy-candidate-metadata.json`
- `raw-policy-generalization-candidate-summary.json`

## Contract

- Load the base candidate checkpoint from
  `outputs/path_feedback_batch_ppo_collector_candidate_v1/`.
- Read collector episodes from
  `outputs/path_feedback_batch_ppo_rollout_collector_dry_run_v1/ppo-rollout-episodes.jsonl`.
- Train only transitions with `ppo_trainable=true` and
  `controlled_choice_source=policy`.
- Never put source-fallback transitions into the optimizer.
- Recompute selected-action `log_prob` and `value` from the base checkpoint;
  mismatch over `1e-4` fails with `ppo_update_not_on_collector_policy`.
- Run full-batch PPO with `seed=0`, `epochs=1`, `learning_rate=1e-5`,
  `clip_ratio=0.2`, `discount_factor=0.99`, and `max_grad_norm=1.0`.
- Advantages are discounted return minus old value.
- Updated checkpoint remains experimental: `publishes_checkpoint=false`,
  `replaces_default_policy=false`, `performance_claimed=false`.

## Closure

1. Refresh upstream sequential evidence under current HEAD:
   `scripts/run_sequential_multi_step_opportunity_closure.sh`.
2. Refresh PPO collector evidence:
   `scripts/run_ppo_rollout_collector_closure.sh`.
3. Run the limited PPO update smoke.
4. Re-evaluate the updated candidate on raw generalization, sequential canary,
   and PPO collector dry-run.
5. Run readiness with `--limited-ppo-update-smoke-summary`.

## Acceptance

- `status=passed`, `reason_codes=[]`.
- `input_ppo_trainable_transition_count>=24`; current re-closure reports 36.
- `optimizer_train_transition_count>=24`; current re-closure reports 36.
- `source_fallback_trainable_count=0`.
- `old_log_prob_max_abs_error<=1e-4`.
- `old_value_max_abs_error<=1e-4`.
- loss, gradients, reward, return, and advantage are finite.
- `parameter_l2_delta>0`.
- `approx_kl<=0.25`.
- `max_grad_norm_after_clip<=1.0`.
- Post-update raw generalization TEST raw regression remains 0.
- Post-update sequential canary keeps 36 episode / 108 step, 6 accepted
  families, `multi_step_accepted_episode_count>=12`, 0 rejected policy choice,
  and all regression gates at 0.
- Post-update collector keeps at least 24 trainable transitions and 0
  invalid/empty mask, missing log-prob/value, non-finite reward, state
  continuity, path/risk, or source-selection regression.
- Readiness reaches `limited_ppo_update_smoke_evaluated` with
  `training_blockers=[]`.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_limited_ppo_update_smoke.py \
  tests/test_ppo_rollout_collector_dry_run.py \
  tests/test_policy_gated_sequential_canary_rollout.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_limited_ppo_update_smoke_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_ppo_collector_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --raw-policy-generalization-evaluation-summary outputs/path_feedback_batch_limited_ppo_update_smoke_v1/raw-policy-generalization-evaluation-summary.json \
  --policy-gated-sequential-canary-rollout-summary outputs/path_feedback_batch_limited_ppo_update_sequential_v1/policy-gated-sequential-canary-rollout-summary.json \
  --ppo-rollout-collector-summary outputs/path_feedback_batch_limited_ppo_update_collector_v1/ppo-rollout-collector-summary.json \
  --limited-ppo-update-smoke-summary outputs/path_feedback_batch_limited_ppo_update_smoke_v1/limited-ppo-update-smoke-summary.json \
  --validate-only
```

## Non-Goals

- No formal PPO rollout.
- No checkpoint publication or default-policy replacement.
- No network, action-space, or default-A* change.
- No distance-contract relaxation.
- No Ackermann-feasible trajectory claim.
- No policy performance claim.
- No IRIS/GCS/path-planner diagnostic promoted to training release evidence.
