# Policy-Gated Sequential Canary Rollout v1

## Summary

Current readiness is `policy_gated_canary_value_stability_evaluated`: the
experimental policy can make safe and better one-step choices across six canary
families. This stage adds a short sequential canary so each next step starts
from the previous controlled execution goal. It remains canary/shadow evidence,
not formal PPO rollout.

## Scope

- Add explicit scenario-spec input to the NPZ validation map generator and
  `scripts/run_path_feedback_validation.sh`.
- Add `configs/policy_gated_sequential_canary_rollout_v1.json`.
- Add `scripts/run_policy_gated_sequential_canary_rollout.py/.sh`.
- Add `scripts/run_policy_gated_sequential_canary_closure.sh`.
- Extend readiness with
  `policy_gated_sequential_canary_rollout_evaluated`.

## Behavior

The runner creates 36 episodes: 6 canary families x 6 variants. Each episode has
3 steps. For step 0, the runner uses the configured initial start cell. For
step N+1, it writes an explicit scenario spec whose `start_cell` is step N's
`controlled_execution_goal_cell`.

If policy selection passes the existing canary gates, the controlled goal is the
policy execution goal. If it fails, the controlled goal is the source-selected
execution goal and the rejection is recorded. State continuity is a hard gate;
independent contexts cannot be grouped and counted as an episode.

## Artifacts

- `outputs/path_feedback_batch_sequential_canary_clean_src_v1/`
- `outputs/path_feedback_batch_sequential_canary_candidate_v1/`
- `outputs/path_feedback_batch_policy_gated_sequential_canary_rollout_v1/`
- `policy-gated-sequential-canary-episodes.jsonl`
- `policy-gated-sequential-canary-steps.jsonl`
- `policy-gated-sequential-canary-rejection-report.json`
- `policy-gated-sequential-canary-rollout-summary.json`

## Acceptance Gates

- `episode_count=36`
- `step_count=108`
- `completed_episode_count>=30`
- `policy_takeover_step_count>=24`
- `accepted_takeover_step_count>=24`
- `accepted_better_step_count>=12`
- `accepted_takeover_family_count=6`
- `multi_step_accepted_episode_count>=12`
- `family_with_multi_step_accepted_episode_count=6`
- `state_continuity_violation_count=0`
- `episode_fallback_count=0`
- `canary_rejected_policy_choice_count=0`
- invalid action mask, fallback/open-grid, safety, contract, path/risk, and
  source-selection regression all 0.
- Readiness becomes
  `policy_gated_sequential_canary_rollout_evaluated` with
  `training_blockers=[]`.

## Verification

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTHON=$P bash scripts/run_policy_gated_sequential_canary_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_sequential_canary_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --raw-policy-generalization-evaluation-summary outputs/path_feedback_batch_sequential_canary_candidate_v1/raw-policy-generalization-evaluation-summary.json \
  --policy-gated-sequential-canary-rollout-summary outputs/path_feedback_batch_policy_gated_sequential_canary_rollout_v1/policy-gated-sequential-canary-rollout-summary.json \
  --validate-only

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_policy_gated_sequential_canary_rollout.py \
  tests/test_policy_training_readiness_review.py \
  tests/test_batch_path_feedback_validation.py

cd dev-platform-constraints && \
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q tests/test_npz_validation_maps.py
```

## Non-Goals

- No formal PPO rollout.
- No PPO parameter update.
- No checkpoint publication or default policy replacement.
- No network/action-space/default-A* change.
- No default distance-contract relaxation.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic-as-training release.
- No policy performance claim.
