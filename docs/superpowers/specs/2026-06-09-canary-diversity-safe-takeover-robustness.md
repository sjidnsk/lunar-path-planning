# Canary Diversity and Safe-Takeover Robustness v1

## Summary

Policy-Gated Canary Rollout v1 proved that the raw policy can safely choose a
source-selected alternative in 2 `npz_mixed_stress_detour` contexts. That proves
the capability exists, but not that it is stable across scenario families. This
stage expands the canary into a multi-family, clean-HEAD, provenance-current
closure.

## Implementation

- Add `policy_canary_diversity` to path-feedback batch and single-run scenario
  selection.
- Add multi-family validation maps in `dev-platform-constraints`, covering
  mixed stress, near-blocked, high-risk tradeoff, dense choke bypass, channel
  contrast, and path-complexity benefit.
- Reuse the existing policy-gated canary evaluator and add per-family summary
  fields: opportunity, changed, source-aligned, accepted, rejected, rejection
  reasons, `accepted_scenario_family_count`,
  `accepted_decision_family_distribution`, and `canary_diversity_passed`.
- Add `scripts/run_canary_diversity_safe_takeover_closure.sh` to refresh
  clean-HEAD SRC, baseline/candidate, raw-generalization, canary diversity, and
  readiness evidence.
- Update readiness so a passing diversity summary advances to
  `policy_gated_canary_diversity_evaluated`.

## Evidence Roots

- SRC: `outputs/path_feedback_batch_canary_diversity_clean_src_v1/`
- CAND: `outputs/path_feedback_batch_canary_diversity_candidate_v1/`
- CANARY: `outputs/path_feedback_batch_policy_gated_canary_diversity_v1/`

## Result

The clean-HEAD closure passed. The canary summary reports
`policy_decision_count=12`, `canary_opportunity_context_count=12`,
`policy_changed_decision_count=6`, `canary_accepted_policy_choice_count=6`,
`canary_rejected_policy_choice_count=0`, `scenario_family_count=6`,
`accepted_scenario_family_count=3`, `canary_diversity_passed=true`, and all
controlled/raw regression gates at `0`. Readiness reports
`training_readiness_status=policy_gated_canary_diversity_evaluated` and
`training_blockers=[]`.

## Acceptance

- No current-git or checkpoint metadata provenance mismatch.
- Diversity batch `failed_count=0`, fallback/open-grid `0`, safety regression
  `0`.
- `context_id_missing_count=0` and `legacy_identity_fallback_count=0`.
- `canary_opportunity_context_count>=12`.
- At least 5 scenario families are present.
- `accepted_scenario_family_count>=3`.
- `policy_changed_decision_count>0`.
- `canary_accepted_policy_choice_count>=4`.
- Accepted choices are not all from the same scenario id or family.
- All rejected choices include reason codes.
- Controlled and raw regression counts are both `0`; invalid action mask,
  fallback, safety, contract, path/risk, and source-selection regression counts
  are all `0`.
- Candidate remains experimental and does not publish a checkpoint, replace the
  default policy, or claim performance.
- Readiness reports `status=passed`,
  `training_readiness_status=policy_gated_canary_diversity_evaluated`, and
  `training_blockers=[]`.

## Non-Goals

This stage does not start formal PPO rollout, publish or replace a policy,
modify network/action space/default A*, relax the distance contract, claim
Ackermann-feasible trajectory, treat IRIS/GCS/path-planner diagnostics as
training release evidence, or claim policy performance improvement.
