# Quasi-Real Guarded PPO Rollout Pilot v1

## Summary

`Quasi-Real Guarded PPO Rollout Pilot v1` moves the training-readiness chain one
step past the quasi-real iterative mini-loop. The rollout entry is now under
test: the policy proposes each step, existing guards decide whether it can
execute, and only train-split, policy-controlled, gate-passed steps become PPO
trainable.

The stage starts only after current-HEAD quasi-real iterative evidence reaches
`iterative_ppo_mini_loop_stability_evaluated` with empty blockers.

## Artifacts

- `configs/guarded_ppo_rollout_pilot_v1.json`
- `configs/guarded_ppo_rollout_update_v1.json`
- `scripts/run_guarded_ppo_rollout_pilot.py/.sh`
- `scripts/run_guarded_ppo_rollout_pilot_closure.sh`
- `outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/`

The output root writes:

- `guarded-ppo-rollout-episodes.jsonl`
- `guarded-ppo-rollout-transitions.jsonl`
- `guarded-ppo-rollout-reward-audit.json`
- `guarded-ppo-rollout-update-summary.json`
- `guarded-ppo-rollout-rejection-report.json`
- `guarded-ppo-rollout-pilot-summary.json`

## Contract

Trainable transitions must satisfy:

- `controlled_choice_source="policy"`
- `ppo_trainable=true`
- `split="train"`
- empty gate reason codes
- valid non-empty action mask
- finite log-prob, value, reward, return, advantage, loss, and gradient
- state continuity valid
- no fallback/open-grid, safety, contract, path/risk, or source-selection
  controlled regression

Source fallback, rejected raw policy probe choices, and validation/test steps
remain diagnostic-only. Raw probe rejection is still counted, but it is not
controlled rollout regression after the gate safely falls back.

## Post-Update Gates

The pilot performs one tiny on-policy PPO update from the same experimental
checkpoint: `seed=0`, `epochs=1`, `learning_rate=1e-5`, `clip_ratio=0.2`,
`discount_factor=0.99`, and `max_grad_norm=1.0`.

After the update it rechecks:

- raw generalization
- generated sequential canary
- generated PPO collector
- quasi-real guarded teacher-following
- quasi-real PPO collector

Passing requires raw TEST regression count 0, generated controlled regression
count 0, quasi-real teacher agreement at least 0.9, and quasi-real collector
trainable transition count at least 24.

## Readiness

Readiness accepts `--guarded-ppo-rollout-pilot-summary` and may advance to
`guarded_ppo_rollout_pilot_evaluated` only when the pilot summary is passed,
reason codes are empty, blockers are empty, and current git provenance matches.

## Non-Goals

This stage does not publish a checkpoint, replace the default policy, expand the
PPO batch, add CUDA/device support, add LOLA or real maps, modify the network,
modify the action space, modify default A*, relax distance/path-risk/source
selection gates, claim Ackermann-feasible trajectory output, treat IRIS/GCS or
path-planner diagnostics as training release evidence, or claim formal training
readiness.
