# Coverage-Driven PPO Improvement Run v1

## Goal

Run the first guarded offline PPO update that consumes the audited coverage-aware reward, then prove or reject exploration-coverage performance improvement against frozen baselines.

## Inputs

- Formal training: `outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1/`
- Post-training replay: `outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1/`
- Selected candidate: `outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1/`
- Coverage signal audit: `outputs/path_feedback_batch_exploration_coverage_signal_audit_v1/`
- Coverage performance evaluation: `outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1/`
- Coverage-aware reward refinement: `outputs/path_feedback_batch_coverage_aware_reward_refinement_v1/`
- Reward source connector: `outputs/path_feedback_batch_connect_reward_component_source_fields_v1/`

## Design

The runner materializes a temporary coverage-aware PPO collector from Stage 4 `reward-component-audit.jsonl` and selected shadow steps. It preserves each row's observation, action mask, controlled action, old `log_prob`, old `value`, reward components, and provenance. The transition reward is `coverage_aware_reward`, not the earlier teacher-following bonus.

The PPO update reuses the existing limited update machinery so old-policy reconstruction, finite loss/return/advantage checks, KL, gradient clipping, parameter delta, training curves, diagnostics, and experimental checkpoint output stay consistent with earlier formal PPO stages. The produced checkpoint is experimental and offline-only.

Post-update evaluation is conservative. It replays the updated checkpoint on the frozen audited observations and credits policy gain only when the updated raw action matches an action with real audited coverage evidence. Guard-rejected or fallback decisions are not counted as PPO coverage improvement. If the updated model remains teacher-equivalent or cannot activate on audited coverage-gain rows, the stage fails with explicit reason codes.

## Outputs

All artifacts are written under `outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1/`:

- `coverage-driven-ppo-improvement-run-summary.json`
- `coverage-aware-ppo-batch/ppo-rollout-episodes.jsonl`
- `coverage-aware-ppo-batch/ppo-rollout-transitions.jsonl`
- `coverage-aware-ppo-batch/ppo-rollout-collector-summary.json`
- `coverage-driven-ppo-update-summary.json`
- `coverage-driven-ppo-training-curves.json`
- `coverage-driven-ppo-diagnostics.json`
- `coverage-driven-experimental-policy-candidate.pt`
- `coverage-driven-experimental-policy-candidate-metadata.json`
- `coverage-driven-ppo-performance-metric-table.jsonl`
- `coverage-driven-ppo-replay-audit.json`
- `coverage-driven-ppo-rejection-report.json`
- `coverage-driven-ppo-improvement-run-report.md`

## Pass And Fail Gates

Pass requires positive `coverage_return_improvement`, positive `valuable_area_covered_improvement`, no coverage-efficiency regression beyond the configured tolerance, `controlled_regression_count=0`, fallback not dominating, finite old `log_prob` and value reconstruction, nonzero parameter delta, provenance clean, and gains across more than one scenario family.

Failure is valid completion for this stage when it is explicit. Expected reason codes include `no_coverage_return_improvement`, `valuable_coverage_not_improved`, `risk_or_cost_regression`, `fallback_dominates`, `guard_regression`, `insufficient_policy_activation`, `coverage_reward_source_not_passed`, and `ppo_update_failed`.

## Boundaries

The stage may create an experimental checkpoint for offline evaluation. It must always keep `publishes_checkpoint=false`, `replaces_default_policy=false`, `performance_claimed=false`, and `formal_release_claimed=false`. It must not connect a real executor, relax guards, change the action space, change network architecture, or modify the default A* planner.

## Current Result And Follow-On Chain

The current run is a clean negative result:

- `status=failed`
- `coverage_driven_ppo_improvement_status=failed`
- `reason_codes=["no_coverage_return_improvement"]`
- `next_required_change=refine_coverage_reward_or_collect_more_policy_coverage`
- `optimizer_train_transition_count=2052`
- `parameter_l2_delta=0.0004370135471177026`
- `coverage_return_improvement=0.0`
- `cumulative_coverage_rate_delta_improvement=0.0`
- `valuable_area_covered_improvement=45.824562342498`
- `accepted_policy_activation_rate=1.0`
- `fallback_rate=0.0`
- `teacher_agreement_rate=1.0`
- `controlled_regression_count=0`

Replay shows that the updated raw policy action, controlled action, and teacher
action are identical for all 2,052 audited rows. The failure is therefore not a
guard rejection problem. The policy update changed parameters but did not change
the accepted action choice away from teacher-equivalent behavior.

The required follow-on chain is:

```text
5A. Policy Coverage Opportunity / Margin Audit
    - expand each context into action-level candidate rows
    - count safe better alternatives versus teacher
    - inspect policy logits/probability margins before and after the update
    - verify candidate-level coverage features available at decision time

5B. Reward-vs-data repair decision
    - if safe better alternatives exist, refine reward/advantage/margin objective
    - if they do not, collect or materialize more policy-differentiating coverage evidence

5C. Rerun Coverage-Driven PPO Improvement Run
    - pass only with positive coverage return and no safety, cost, risk, or fallback regression
```

Do not proceed to shadow/canary release validation until Stage 5C passes.
