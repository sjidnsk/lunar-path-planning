# PPO Rollout Contract and Collector Dry-Run v1

## Background

The project has reached `policy_gated_sequential_multi_step_opportunity_evaluated`.
The policy-gated sequential canary completes 36 episodes / 108 steps with 37
accepted better takeovers, 6-family coverage, no rejected policy choice, no
state-continuity violation, and zero cumulative safety/contract/path/risk/source
selection regression.

The next boundary is not formal PPO training. It is a collector contract dry-run:
prove that accepted policy-controlled sequential steps can become existing
`RolloutEpisode/RolloutTransition` records with valid action masks, log-prob,
value, reward, metadata, and dataset validation.

## Implemented Interfaces

- `configs/sequential_evidence_consistency_v1.json`
- `scripts/run_sequential_evidence_consistency_check.py/.sh`
- `configs/ppo_rollout_collector_dry_run_v1.json`
- `scripts/run_ppo_rollout_collector_dry_run.py/.sh`
- `scripts/run_ppo_rollout_collector_closure.sh`
- readiness status: `ppo_rollout_collector_dry_run_evaluated`
- readiness input: `--ppo-rollout-collector-summary`

The collector writes:

- `ppo-rollout-episodes.jsonl`
- `ppo-rollout-transitions.jsonl`
- `ppo-rollout-collector-summary.json`
- `ppo-rollout-rejection-report.json`
- `ppo-rollout-reward-audit.json`

## Contract

Only accepted policy-controlled steps are PPO-trainable:

- `controlled_choice_source=policy`
- `canary_gate_passed=true`
- no canary/raw/controlled regression reason
- valid action mask
- finite reward
- present log-prob and value

`source_fallback` steps are diagnostic-only and must not become on-policy PPO
positive samples. The collector uses the existing rollout schema and validates
materialized episodes with `validate_rollout_dataset`.

## Current Evidence

Roots:

- upstream SRC: `outputs/path_feedback_batch_ppo_collector_clean_src_v1/`
- upstream CAND: `outputs/path_feedback_batch_ppo_collector_candidate_v1/`
- upstream SEQ: `outputs/path_feedback_batch_ppo_collector_sequential_v1/`
- collector: `outputs/path_feedback_batch_ppo_rollout_collector_dry_run_v1/`

Current clean-HEAD closure result:

- `status=passed`
- `reason_codes=[]`
- `episode_count=36`
- `step_count=108`
- `materialized_episode_count=19`
- `ppo_trainable_transition_count=37`
- `diagnostic_transition_count=71`
- `source_fallback_trainable_count=0`
- `invalid_action_mask_count=0`
- `empty_action_mask_count=0`
- `missing_log_prob_count=0`
- `missing_value_count=0`
- `non_finite_reward_count=0`
- `state_continuity_violation_count=0`
- path/risk/source-selection regression counts are all `0`
- readiness `training_readiness_status=ppo_rollout_collector_dry_run_evaluated`
- readiness `training_blockers=[]`

The collector summary includes git provenance and the closure refreshes upstream
sequential SRC/CAND/SEQ before materialization. A final readiness
`--validate-only` run must remain part of the acceptance gate whenever tracked
files change.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTHON=$P bash scripts/run_sequential_multi_step_opportunity_closure.sh

PYTHON=$P bash scripts/run_ppo_rollout_collector_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_ppo_collector_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --raw-policy-generalization-evaluation-summary outputs/path_feedback_batch_ppo_collector_candidate_v1/raw-policy-generalization-evaluation-summary.json \
  --policy-gated-sequential-canary-rollout-summary outputs/path_feedback_batch_ppo_collector_sequential_v1/policy-gated-sequential-canary-rollout-summary.json \
  --ppo-rollout-collector-summary outputs/path_feedback_batch_ppo_rollout_collector_dry_run_v1/ppo-rollout-collector-summary.json \
  --validate-only

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_ppo_rollout_collector_dry_run.py \
  tests/test_policy_gated_sequential_canary_rollout.py \
  tests/test_policy_training_readiness_review.py
```

## Non-Goals

- No formal PPO rollout.
- No PPO optimizer update.
- No checkpoint publication or default policy replacement.
- No network/action-space/default-A* change.
- No distance-contract relaxation.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic-as-training release.
- No policy performance claim.
