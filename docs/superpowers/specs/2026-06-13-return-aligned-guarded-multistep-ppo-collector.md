# Return-Aligned Guarded Multi-Step PPO Collector Expansion v1

## Summary

This stage expands guarded PPO collector evidence from clean one-step
transitions into auditable multi-step return episodes. It does not run a new PPO
update. The main line remains teacher-skill learning, while better-than-teacher
evidence is allowed only through multi-step return accounting. A better choice
does not need to differ from the teacher.

## Inputs

- `outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/`
- `outputs/path_feedback_batch_guarded_ppo_evidence_freeze_v1/guarded-ppo-evidence-freeze-summary.json`
- `outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/pilot/collector/ppo-rollout-transitions.jsonl`

The runner first verifies that the guarded pilot and evidence freeze summaries
passed and that source git provenance matches the current HEAD.

## Artifacts

- `configs/return_aligned_guarded_multi_step_ppo_collector_expansion_v1.json`
- `scripts/run_return_aligned_guarded_multi_step_ppo_collector_expansion.py/.sh`
- `scripts/run_return_aligned_guarded_multi_step_ppo_collector_closure.sh`
- `outputs/path_feedback_batch_return_aligned_guarded_multi_step_ppo_collector_expansion_v1/return-aligned-ppo-episodes.jsonl`
- `outputs/path_feedback_batch_return_aligned_guarded_multi_step_ppo_collector_expansion_v1/return-aligned-ppo-transitions.jsonl`
- `outputs/path_feedback_batch_return_aligned_guarded_multi_step_ppo_collector_expansion_v1/return-aligned-reward-audit.json`
- `outputs/path_feedback_batch_return_aligned_guarded_multi_step_ppo_collector_expansion_v1/return-aligned-rejection-report.json`
- `outputs/path_feedback_batch_return_aligned_guarded_multi_step_ppo_collector_expansion_v1/return-aligned-collector-summary.json`

## Contract

Default horizon is 3. Step-level PPO trainability remains strict:

- `split=train`
- `controlled_choice_source=policy`
- input collector has `ppo_trainable=true`
- gate reason codes are empty
- reward is finite

Validation/test split, source fallback, teacher fallback, none/not_scored, raw
rejected policy probes, and non-empty gate reasons stay diagnostic-only.

Episode-level trainability is a return-audit contract. A full-horizon train
episode is trainable when it has finite discounted return and advantage, no
source fallback, no gate diagnostic reason, and no controlled
safety/contract/path-risk/source-selection regression.

Reward audit fields:

- `teacher_following_return`
- `teacher_equivalent_return`
- `safe_better_return`
- `controlled_regression_penalty`
- `discounted_episode_return`
- `advantage_reference_value`

The audit must declare `uses_multistep_discounted_return=true` and
`not_single_step_best_action=true`.

## Current Result

The closure passes:

- `status=passed`
- `reason_codes=[]`
- `horizon=3`
- `episode_count=36`
- `step_count=108`
- `trainable_episode_count=31`
- `trainable_transition_count=30`
- `diagnostic_transition_count=78`
- `source_fallback_trainable_count=0`
- `non_finite_reward_count=0`
- `non_finite_return_count=0`
- `non_finite_advantage_count=0`
- `controlled_regression_count=0`
- readiness status:
  `return_aligned_guarded_multistep_collector_evaluated`

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_return_aligned_guarded_multi_step_ppo_collector_expansion.py \
  tests/test_guarded_ppo_evidence_freeze.py \
  tests/test_guarded_ppo_rollout_pilot.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_return_aligned_guarded_multi_step_ppo_collector_closure.sh

jq '{status, reason_codes, horizon, trainable_episode_count, trainable_transition_count, source_fallback_trainable_count, controlled_regression_count}' \
  outputs/path_feedback_batch_return_aligned_guarded_multi_step_ppo_collector_expansion_v1/return-aligned-collector-summary.json

git diff --check
```

## Non-Goals

- No formal PPO training
- No new PPO update
- No checkpoint publication
- No default policy replacement
- No network/action-space/default-A* change
- No safety/contract/path-risk/source-selection gate relaxation
- No Ackermann-feasible trajectory claim
- No IRIS/GCS/path-planner diagnostic as training release evidence
- No policy performance or formal training-ready claim
