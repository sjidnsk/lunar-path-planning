# Guarded Formal PPO Post-Training Stability Replay v1

## Summary

`Guarded Formal PPO Training Run v1` has produced five experimental candidates
from 684 authorized quasi-real trainable transitions. This stage validates those
already-trained candidates by replaying post-training gates. It does not run a
new PPO update, publish a checkpoint, or replace the default policy.

## Inputs

- Formal training run summary:
  `outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1/formal-ppo-training-run-summary.json`
- Formal training seed summaries:
  `outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1/formal-ppo-training-run-seed-summaries.jsonl`
- Per-seed checkpoints:
  `outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1/seed-XX/limited_ppo_update_smoke/experimental-hybrid-policy-candidate.pt`

The input formal training run must be `status=passed`, have empty
`reason_codes`, report clean/current git provenance, use five passed seeds, and
remain experimental only.

## Outputs

- `configs/guarded_formal_ppo_post_training_stability_replay_v1.json`
- `scripts/run_guarded_formal_ppo_post_training_stability_replay.py/.sh`
- `scripts/run_guarded_formal_ppo_post_training_stability_replay_closure.sh`
- `tests/test_guarded_formal_ppo_post_training_stability_replay.py`
- `outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1/formal-ppo-post-training-stability-replay-summary.json`
- `formal-ppo-post-training-stability-replay-seed-summaries.jsonl`
- `formal-ppo-post-training-stability-replay-progress.jsonl`
- `formal-ppo-post-training-stability-replay-drift-report.jsonl`
- `formal-ppo-post-training-stability-replay-gate-audit.json`
- `formal-ppo-post-training-stability-replay-rollback-manifest.json`
- `formal-ppo-post-training-stability-replay-readiness-validate-only.json`
- `formal-ppo-post-training-stability-replay-report.md`

Documentation updates are required in `README.md` and
`docs/算法设计与系统架构报告.md`.

## Acceptance Gates

- Summary `status=passed`, `reason_codes=[]`.
- `seed_count=5`, `replay_count_per_seed>=3`.
- `total_replay_count>=15`, `passed_replay_count=total_replay_count`.
- `missing_seed_candidate_checkpoint_count=0`.
- `replay_behavior_drift_count=0`.
- `optimizer_train_transition_count=684`.
- `replay_collector_trainable_transition_count>=684`.
- validation/test/fallback/diagnostic trainable counts are all zero.
- missing observation/log_prob/value counts are zero.
- non-finite reward/return/advantage counts are zero.
- `teacher_agreement_rate>=0.95`.
- controlled safety/contract/path-risk/source-selection regression counts are zero.
- holdout/canary replay gates are passed.
- `publishes_checkpoint=false`, `replaces_default_policy=false`.
- `performance_claimed=false`, `formal_training_ready_claimed=false`.
- Readiness validate-only returns
  `guarded_formal_ppo_post_training_stability_replay_evaluated` with no
  blockers or reason codes.

## Verification

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_guarded_formal_ppo_post_training_stability_replay.py \
  tests/test_guarded_formal_ppo_training_run.py \
  tests/test_policy_training_readiness_review.py
PYTHON=$P bash scripts/run_guarded_formal_ppo_post_training_stability_replay_closure.sh
PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --guarded-formal-ppo-post-training-stability-replay-summary \
    outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1/formal-ppo-post-training-stability-replay-summary.json \
  --validate-only
git diff --check
```

## Non-Goals

- Do not run a new PPO update.
- Do not start an online canary or connect a real executor.
- Do not publish or replace a checkpoint.
- Do not replace the default policy.
- Do not modify network, action space, or default A*.
- Do not relax distance/path-risk/source-selection gates.
- Do not download new raw data.
- Do not claim Ackermann-feasible trajectory.
- Do not treat IRIS/GCS/path-planner diagnostics as training or release gates.
- Do not claim policy performance improvement or deployability.
