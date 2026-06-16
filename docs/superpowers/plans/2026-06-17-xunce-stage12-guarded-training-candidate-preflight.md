# Xunce Stage 12 Guarded Training Candidate Preflight v1 Plan

## Goal

Stage 12 converts the completed research-network evidence into a guarded authorization preflight for the next controlled training candidate stage. It verifies that Stage 8-11 evidence is present, passed, clean, and boundary-safe before any PPO update is allowed in Stage 13.

This stage does not train PPO, write or publish checkpoints, replace default policy, connect an executor, or claim performance.

## Inputs

- Stage 8 full network summary.
- Stage 9 static contract summary.
- Stage 10 ablation summary.
- Stage 11 stress summary.

## Interface

Add:

- `configs/xunce_guarded_training_candidate_preflight_v1.json`
- `scripts/run_xunce_guarded_training_candidate_preflight.py`
- `scripts/run_xunce_guarded_training_candidate_preflight.sh`
- `tests/test_xunce_guarded_training_candidate_preflight.py`

Default output root:

`outputs/path_feedback_batch_xunce_guarded_training_candidate_preflight_v1/`

Artifacts:

- `xunce-guarded-training-candidate-preflight-summary.json`
- `xunce-guarded-training-candidate-preflight-manifest.json`
- `xunce-guarded-training-source-evidence-audit.json`
- `xunce-guarded-training-boundary-audit.json`
- `xunce-guarded-training-rejection-report.json`
- `xunce-guarded-training-candidate-preflight-report.md`

## Decision Contract

Summary fields:

- `status`
- `reason_codes`
- `source_full_network_status`
- `source_static_contract_status`
- `source_ablation_status`
- `source_stress_status`
- `architecture=xunce_full_network_v1`
- `full_network_v1_passed`
- `static_contract_validation_passed`
- `ablation_experiments_passed`
- `stress_evaluation_passed`
- `training_candidate_preflight_passed`
- `training_preflight_verdict`
- `controlled_training_candidate_authorized`
- `next_required_change`

Boundary fields must stay false:

- `runs_new_ppo_update`
- `publishes_checkpoint`
- `replaces_default_policy`
- `connects_real_executor`
- `starts_online_canary`
- `modifies_action_space`
- `modifies_default_astar`
- `real_world_release_approved`
- `real_world_performance_claimed`

Passing next gate:

`controlled_training_candidate`

Failure next gate:

`fix_guarded_training_candidate_preflight`

If Stage 11 evidence is missing or not passed:

`fix_full_network_stress_evaluation`

## Verification

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_guarded_training_candidate_preflight.py -q
PYTHON=$PY bash scripts/run_xunce_guarded_training_candidate_preflight.sh
jq '{status,reason_codes,training_candidate_preflight_passed,controlled_training_candidate_authorized,next_required_change,publishes_checkpoint,runs_new_ppo_update}' \
  outputs/path_feedback_batch_xunce_guarded_training_candidate_preflight_v1/xunce-guarded-training-candidate-preflight-summary.json
rg -n "Xunce Guarded Training Candidate Preflight v1|run_xunce_guarded_training_candidate_preflight|controlled_training_candidate" \
  README.md docs/算法设计与系统架构报告.md docs/superpowers/specs
git diff --check
```

## Non-goals

- No PPO update.
- No checkpoint write or publication.
- No default-policy replacement.
- No executor connection.
- No online canary.
- No performance or real-world claim.
