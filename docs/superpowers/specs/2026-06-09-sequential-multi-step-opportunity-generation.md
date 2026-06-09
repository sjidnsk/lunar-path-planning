# Sequential Multi-Step Opportunity Generation v1

## Background

Current sequential safe-choice evidence proves that policy-induced short
sequences can be safe after calibration, but not that multi-step takeover
opportunities are distributed broadly enough. Latest blocker:

- `episode_count=36`
- `step_count=108`
- `accepted_better_step_count=28`
- `accepted_takeover_family_count=6`
- `canary_rejected_policy_choice_count=0`
- cumulative safety/contract/path/risk/source-selection regression all `0`
- `multi_step_accepted_episode_count=6 < 12`
- `family_with_multi_step_accepted_episode_count=2 < 6`
- `next_required_change=sequential_opportunity_distribution_gap_requires_more_episodes`

The next step is therefore opportunity generation and measurement, not gate
relaxation and not formal PPO rollout.

## Objective

Create a scenario-distributed sequential canary closure that first proves every
family has real consecutive `gate-safe + better` alternatives, then evaluates
whether the policy accepts those opportunities under the unchanged strict gate.

## Scope

- Add `policy_canary_sequential_multi_step_opportunity` scenario set in
  `dev-platform-constraints/scripts/generate_npz_validation_maps.py`.
- Extend sequential runner config support for `scenario_set` and
  `template_scenario_id_prefix`.
- Add opportunity diagnosis:
  - `scripts/run_sequential_multi_step_opportunity_diagnosis.py/.sh`
  - `configs/sequential_multi_step_opportunity_diagnosis_v1.json`
- Add rollout/static configs:
  - `configs/policy_gated_sequential_multi_step_opportunity_rollout_v1.json`
  - `configs/path_feedback_batch_sequential_multi_step_opportunity_v1.json`
- Add closure:
  - `scripts/run_sequential_multi_step_opportunity_closure.sh`
- Extend readiness status:
  - `policy_gated_sequential_multi_step_opportunity_evaluated`
- Update docs:
  - `README.md`
  - `docs/算法设计与系统架构报告.md`

## Outputs

- `outputs/path_feedback_batch_sequential_multi_step_opportunity_clean_src_v1/`
- `outputs/path_feedback_batch_sequential_multi_step_opportunity_candidate_v1/`
- `outputs/path_feedback_batch_policy_gated_sequential_multi_step_opportunity_preflight_v1/`
- `outputs/path_feedback_batch_policy_gated_sequential_multi_step_opportunity_rollout_v1/`
- `sequential-multi-step-opportunity-diagnosis-summary.json`
- `sequential-multi-step-opportunity-diagnostics.jsonl`
- `sequential-multi-step-opportunity-exclusion-report.json`

## Acceptance

Opportunity diagnosis:

- `episode_count=36`
- `step_count=108`
- `multi_step_opportunity_episode_count>=12` using required step indexes `[0, 1]`
- `min_multi_step_opportunity_episode_count_per_family>=2`
- `family_with_multi_step_opportunity_count=6`
- `safe_better_alternative_step_count>=24`
- `opportunity_exclusion_count=0`

Final sequential rollout:

- `completed_episode_count=36`
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
  source-selection regression all `0`

Readiness:

- `status=passed`
- `training_readiness_status=policy_gated_sequential_multi_step_opportunity_evaluated`
- `training_blockers=[]`

If diagnosis fails, fix scenario/opportunity generation. If diagnosis passes
but policy misses existing opportunities, use missed-safe-choice preference
calibration and rerun the same gate.

## Current Result

The repaired closure now passes when run from the committed repair. The closure
was split into a preflight opportunity root and a final calibrated rollout root
so that scenario opportunity measurement is not invalidated by the policy
changing later episode states.

Preflight diagnosis:

- root:
  `outputs/path_feedback_batch_policy_gated_sequential_multi_step_opportunity_preflight_v1/`
- `status=passed`, `reason_codes=[]`
- `episode_count=36`, `step_count=108`
- `safe_better_alternative_step_count=54`
- `multi_step_opportunity_episode_count=15`
- `family_with_multi_step_opportunity_count=6`
- per-family multi-step opportunity episodes:
  `channel_contrast=3`, `dense_choke_safe_bypass=2`,
  `high_risk_tradeoff=3`, `mixed_stress_detour=3`,
  `near_blocked_safe_alt=2`, `path_complexity_benefit=2`

Final calibrated sequential rollout:

- root:
  `outputs/path_feedback_batch_policy_gated_sequential_multi_step_opportunity_rollout_v1/`
- `status=passed`, `reason_codes=[]`
- `episode_count=36`, `step_count=108`
- `policy_takeover_step_count=37`
- `accepted_takeover_step_count=37`
- `accepted_better_step_count=37`
- `accepted_takeover_family_count=6`
- `multi_step_accepted_episode_count=12`
- `family_with_multi_step_accepted_episode_count=6`
- `canary_rejected_policy_choice_count=0`
- `invalid_action_mask_count=0`
- cumulative path/risk regression counts are `0`

Readiness reaches
`training_readiness_status=policy_gated_sequential_multi_step_opportunity_evaluated`
with `training_blockers=[]` when the closure is refreshed from the current
committed HEAD.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTHON=$P bash scripts/run_sequential_multi_step_opportunity_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_sequential_multi_step_opportunity_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --raw-policy-generalization-evaluation-summary outputs/path_feedback_batch_sequential_multi_step_opportunity_candidate_v1/raw-policy-generalization-evaluation-summary.json \
  --policy-gated-sequential-canary-rollout-summary outputs/path_feedback_batch_policy_gated_sequential_multi_step_opportunity_rollout_v1/policy-gated-sequential-canary-rollout-summary.json \
  --validate-only

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_sequential_multi_step_opportunity_generation.py \
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
