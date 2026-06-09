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

Root:
`outputs/path_feedback_batch_ppo_rollout_collector_dry_run_v1/`

Current dry-run result:

- `status=passed`
- `reason_codes=[]`
- `episode_count=36`
- `step_count=108`
- `ppo_trainable_transition_count=37`
- `source_fallback_trainable_count=0`
- `invalid_action_mask_count=0`
- `empty_action_mask_count=0`
- `missing_log_prob_count=0`
- `missing_value_count=0`
- `non_finite_reward_count=0`

The collector summary now includes git provenance. Because this feature changes
tracked files, full readiness validate-only still needs a clean-HEAD evidence
refresh after commit; old SRC/CAND/SEQ roots correctly report
`current_git_provenance_mismatch` while the worktree is dirty.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTHON=$P bash scripts/run_sequential_evidence_consistency_check.sh \
  --batch-root outputs/path_feedback_batch_policy_gated_sequential_multi_step_opportunity_rollout_v1 \
  --readiness-summary outputs/path_feedback_batch_sequential_multi_step_opportunity_clean_src_v1/policy-training-readiness-review-summary.json \
  --config configs/sequential_evidence_consistency_v1.json

PYTHON=$P bash scripts/run_ppo_rollout_collector_dry_run.sh \
  --sequential-root outputs/path_feedback_batch_policy_gated_sequential_multi_step_opportunity_rollout_v1 \
  --candidate-root outputs/path_feedback_batch_sequential_multi_step_opportunity_candidate_v1 \
  --output-root outputs/path_feedback_batch_ppo_rollout_collector_dry_run_v1 \
  --config configs/ppo_rollout_collector_dry_run_v1.json

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
