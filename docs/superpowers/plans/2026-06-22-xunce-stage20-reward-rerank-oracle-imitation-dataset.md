# Stage 20 Reward-Rerank Oracle Imitation Dataset

## Goal

Stage 20 turns Stage 19 reward-rerank oracle preference evidence into strict
teacher-label samples for Xunce and runs a small teacher-imitation dry-run. It
does not run PPO, publish a checkpoint, replace the default policy, connect a
real executor, or start canary traffic.

## Inputs

- Stage 19 root:
  `D:\CodexDownloads\lunar-path-planning\stage19_evaluator_critic_preflight\outputs\path_feedback_batch_xunce_stage19_evaluator_critic_preflight_v1`
- Required Stage 19 artifacts:
  - `xunce-stage19-evaluator-critic-preflight-summary.json`
  - `xunce-stage19-practical-target-selection.json`
  - `xunce-stage19-critic-target-readiness.json`
  - `xunce-stage19-preference-pair-audit.jsonl`

## Trainable Sample Rules

A preference row becomes a teacher sample only when all conditions hold:

- `baseline_policy=xunce`
- `same_candidate_set=true`
- `hard_risk_clean_pair=true`
- oracle and Xunce selected different action indices
- oracle and Xunce share the same `candidate_set_hash`

Rows from different trajectories, different candidate sets, incumbent-only
baselines, hard-risk-unclean pairs, and rows where oracle is not better by the
coverage/cost rule are written to the exclusion report.

## Outputs

Default output root:

```text
D:\CodexDownloads\lunar-path-planning\stage20_reward_rerank_oracle_imitation_dataset\outputs\path_feedback_batch_xunce_stage20_reward_rerank_oracle_imitation_dataset_v1
```

Artifacts:

- `xunce-stage20-oracle-imitation-summary.json`
- `xunce-stage20-oracle-imitation-teacher-samples.jsonl`
- `xunce-stage20-oracle-imitation-exclusion-report.jsonl`
- `xunce-stage20-oracle-imitation-dataset-stats.json`
- `xunce-stage20-oracle-imitation-dry-run-summary.json`
- `xunce-stage20-next-stage-routing.json`
- `xunce-stage20-report.md`
- `xunce-stage20-manifest.json`

## Routing

- Fewer than 24 trainable pairs:
  `collect_more_reward_rerank_same_candidate_preference_evidence`, dry-run
  skipped.
- 24 to 199 trainable pairs:
  dry-run may execute, but route remains
  `collect_more_reward_rerank_same_candidate_preference_evidence`.
- 200 or more trainable pairs with passing dry-run:
  `stage20_1_supervised_oracle_imitation_checkpoint_preflight`.

Every route keeps:

```text
stage20_authorized=false
runs_new_ppo_update=false
publishes_checkpoint=false
replaces_default_policy=false
connects_real_executor=false
starts_online_canary=false
canary_traffic_fraction=0.0
```

## Validation

```powershell
python -m pytest tests\test_xunce_stage20_reward_rerank_oracle_imitation_dataset.py tests\test_xunce_stage18_research_evidence_pipeline.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage20-oracle-imitation
python -m py_compile scripts\run_xunce_stage20_reward_rerank_oracle_imitation_dataset.py scripts\xunce_stage18_pipeline.py scripts\run_xunce_stage18_research_evidence_pipeline.py
python scripts\run_stage.py --stage xunce-stage20-reward-rerank-oracle-imitation-dataset --dry-run
```

## Current Expected Outcome

Stage 19 currently contains many preference audit rows, but only a small strict
same-candidate Xunce-vs-oracle subset. Stage 20 is expected to validate the data
pipeline and teacher-imitation loss, then route to collecting more
same-candidate oracle imitation evidence before any checkpoint-training
preflight.
