# Xunce Stage 19 Evaluator / Critic Preflight Plan

## Summary

Stage 19 turns the passed Stage 18.11 path-cost calibration evidence into a
human-review-only evaluator/critic preflight. It does not train Xunce, publish a
checkpoint, replace the default policy, connect a real executor, or start canary
traffic.

The practical target is the diagnostic reward-rerank oracle with
`candidate_count=36` and `path_cost_weight=0.1`. The `path_cost_weight=0.0`
diagnostic rollout remains an upper-bound coverage reference only because it
uses much higher path cost.

## Inputs

- Stage 18.11 calibration root:
  `D:\CodexDownloads\lunar-path-planning\stage18_11_path_cost_weight_calibration\outputs\path_feedback_batch_xunce_stage18_11_path_cost_weight_calibration_v1`
- Stage 18.11 diagnostic rollout roots referenced by the calibration summary.
- Canonical profile v3 lineage.
- Target capped final coverage: `0.99`.

## Runner

- Script: `scripts/run_xunce_stage19_evaluator_critic_preflight.py`
- Config: `configs/xunce_stage19_evaluator_critic_preflight_v1.json`
- Stage registry id: `xunce-stage19-evaluator-critic-preflight`
- Default output root:
  `D:\CodexDownloads\lunar-path-planning\stage19_evaluator_critic_preflight\outputs\path_feedback_batch_xunce_stage19_evaluator_critic_preflight_v1`

## Outputs

- `xunce-stage19-evaluator-critic-preflight-summary.json`
- `xunce-stage19-diagnostic-rollout-evaluator.json`
- `xunce-stage19-practical-target-selection.json`
- `xunce-stage19-preference-pair-audit.jsonl`
- `xunce-stage19-critic-target-readiness.json`
- `xunce-stage19-next-stage-routing.json`
- `xunce-stage19-report.md`
- `xunce-stage19-manifest.json`

## Decision Rules

- Missing or invalid Stage 18.11 inputs route to
  `rerun_stage18_11_reward_rerank_diagnostic_rollouts`.
- Hard risk violations route to `repair_path_risk_boundary_filtering`.
- No diagnostic rollout reaching capped final coverage >= `0.99` routes to
  `stage18_12_rollout_horizon_or_mission_budget_scaling_for_99pct_coverage`.
- If only the high-cost upper-bound target is feasible, route to
  `continue_path_cost_weight_calibration_at_99pct_coverage`.
- If preference evidence is not ready, route to
  `collect_more_reward_rerank_preference_evidence`.
- If the practical target is feasible and critic evidence is ready, route to
  `stage20_reward_rerank_oracle_preference_dataset_preparation`.

Every route keeps:

- `stage20_authorized=false`
- `training_or_release_authorized=false`
- `runs_new_ppo_update=false`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `connects_real_executor=false`
- `starts_online_canary=false`
- `canary_traffic_fraction=0.0`

## Current Result

The Stage 19 run passed with:

- selected practical target: `candidate_count=36`, `path_cost_weight=0.1`
- oracle capped final coverage mean: about `0.9998`
- oracle mean path cost: about `674m`
- hard risk violation count: `0`
- fixed Xunce checkpoint advantage: `false`
- next route: `stage20_reward_rerank_oracle_preference_dataset_preparation`

This means the oracle behavior is a useful evaluator/critic target, but it is
not yet a trained Xunce model capability.

## Verification

```powershell
python -m pytest tests\test_xunce_stage19_evaluator_critic_preflight.py tests\test_xunce_stage18_research_evidence_pipeline.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage19-final
python -m py_compile scripts\run_xunce_stage19_evaluator_critic_preflight.py scripts\xunce_stage18_pipeline.py scripts\run_xunce_stage18_research_evidence_pipeline.py
python scripts\run_stage.py --stage xunce-stage19-evaluator-critic-preflight --dry-run
python scripts\run_xunce_stage19_evaluator_critic_preflight.py --config configs\xunce_stage19_evaluator_critic_preflight_v1.json --stage18-11-path-cost-weight-calibration-root D:\CodexDownloads\lunar-path-planning\stage18_11_path_cost_weight_calibration\outputs\path_feedback_batch_xunce_stage18_11_path_cost_weight_calibration_v1 --output-root D:\CodexDownloads\lunar-path-planning\stage19_evaluator_critic_preflight\outputs\path_feedback_batch_xunce_stage19_evaluator_critic_preflight_v1 --repo-root .
```
