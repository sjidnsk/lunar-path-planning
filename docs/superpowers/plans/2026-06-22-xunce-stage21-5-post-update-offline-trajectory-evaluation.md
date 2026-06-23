# Stage 21.5 Post-Update Offline Trajectory Evaluation Plan

## Summary

Stage 21.5 验证 Stage 21.4 tiny PPO update 后，Xunce 在同一小规模高保真离线探索任务中是否出现可观察的轨迹退化或初步改进信号。本阶段不再看 loss 是否下降，而是看整场 trajectory：final coverage、coverage AUC、new cells、path cost、coverage per 100m、hard risk、soft risk、model inference failure、mask violation、unreachable/path-planning/open-grid 边界和 candidate exhaustion。

Stage 21.5 仍是离线评估，不训练、不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。

## Inputs

- Stage21.4 root:
  `D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_4_tiny_ppo_update_smoke_v1`
- Pre-PPO checkpoint: Stage21.4 summary 中的 `source_xunce_candidate_checkpoint`
- Post-PPO checkpoint: Stage21.4 summary 中的 `experimental_checkpoint_path`
- High-fidelity config:
  `configs/xunce_high_fidelity_exploration_coverage_comparison_stage18_9_strict_v3.json`

## Default Smoke Scope

默认使用小规模 smoke，避免直接扩大预算：

- `required_scenario_count=2`
- `rollout_steps=4`
- `dynamic_max_candidates_per_step=36`
- `dynamic_proposal_pool_limit_per_step=288`
- `include_oracle_baselines=true`
- `include_roi_weighted_coverage=true`
- `include_canonical_reward_rerank_oracle=true`
- `canonical_reward_rerank_profile=configs/xunce_canonical_reward_guard_profile_v3_path_cost_w010.json`

扩大到 24 scenarios x 40 steps、更长 horizon、更多 seed 或更大 candidate pool 前，需要用户确认预算。

## Implementation

新增 runner:

`scripts/run_xunce_stage21_5_post_update_offline_trajectory_evaluation.py`

新增 config:

`configs/xunce_stage21_5_post_update_offline_trajectory_evaluation_v1.json`

runner 行为：

1. 校验 Stage21.4 summary/routing/checkpoint audit 通过，post checkpoint 可重载且 `experimental_only=true`。
2. 接受 Stage21.4 旧 route `implement_stage21_5_single_seed_ppo_pilot` 作为兼容别名，也接受显式 route `implement_stage21_5_post_update_offline_trajectory_evaluation`。
3. 用同一 high-fidelity 配置跑两组评估：
   - `pre_ppo_xunce`: `xunce_candidate_checkpoint = source_xunce_candidate_checkpoint`
   - `post_ppo_xunce`: `xunce_candidate_checkpoint = experimental_checkpoint_path`
4. 每组保留 incumbent、greedy coverage oracle、cost-aware coverage oracle、reward-rerank oracle 作为诊断视角。
5. 只从 `xunce-exploration-coverage-episodes.jsonl` 读取 `policy == "xunce"` 的行，按 `scenario_id` 一对一 join，计算 pre/post delta。
6. 不使用 high-fidelity summary 中的 `xunce_*_delta_vs_incumbent` 字段作为 pre/post 证据。
7. 输出 summary、policy evaluation results、trajectory delta、routing、report、manifest。

为测试可控，runner 支持 `execute_high_fidelity_evaluations=false` 并消费已有 pre/post evaluation roots；真实 smoke 使用 `true`。

## Outputs

默认输出根：

`D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_5_post_update_offline_trajectory_evaluation_v1`

产物：

- `xunce-stage21-5-post-update-evaluation-summary.json`
- `xunce-stage21-5-policy-evaluation-results.json`
- `xunce-stage21-5-trajectory-delta.json`
- `xunce-stage21-5-scenario-trajectory-delta.jsonl`
- `xunce-stage21-5-next-stage-routing.json`
- `xunce-stage21-5-report.md`
- `xunce-stage21-5-manifest.json`
- `pre_ppo_xunce/` high-fidelity artifacts
- `post_ppo_xunce/` high-fidelity artifacts

## Decision Rules

- 输入缺失或 Stage21.4 未通过：
  `rerun_stage21_5_required_inputs`
- 任一评估 checkpoint/load/model inference 失败，scenario/rollout 不匹配，或 pre/post Xunce scenario id 不一致：
  `repair_stage21_5_offline_evaluation_execution`
- post-PPO 出现 hard risk、mask violation、unreachable selected candidate、path-planning failure、open-grid fallback、model inference failure：
  `repair_stage21_5_hard_risk_regression`
- post-PPO final coverage、capped final coverage、coverage AUC、capped coverage AUC 任一低于 pre-PPO：
  `repair_stage21_5_reward_collector_advantage_or_horizon`
- 所有覆盖指标不退化，post hard risk 和执行边界均为 0：
  `implement_stage21_6_multi_seed_ppo_pilot`

注意：Stage21.4 只有 8 条训练样本，因此 Stage21.5 即使通过，也只能说明 tiny PPO update 没有破坏离线 smoke，并可能有初步改进信号；不能宣称模型可发布。

## Tests

新增：

`tests/test_xunce_stage21_5_post_update_offline_trajectory_evaluation.py`

覆盖：

- 缺 Stage21.4 或 checkpoint audit 失败时 hard fail。
- 可消费 fake pre/post evaluation roots 并计算 delta。
- Stage21.4 route 兼容别名可通过。
- post coverage/AUC 均不退化且 hard risk=0 时 route 到 Stage21.6。
- final coverage 退化但 AUC 持平时不能放行。
- AUC 退化但 final coverage 持平时不能放行。
- post hard risk violation >0 时 route 到 hard-risk repair。
- pre/post scenario id 不一致时 route 到 execution repair。
- 所有发布/替换/executor/canary 字段必须 false，`runs_new_ppo_update=false`。
- registry dry-run 支持新 stage。

## Verification

```powershell
python -m pytest tests\test_xunce_stage21_5_post_update_offline_trajectory_evaluation.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage21-5
python -m py_compile scripts\run_xunce_stage21_5_post_update_offline_trajectory_evaluation.py
python scripts\run_stage.py --stage xunce-stage21-5-post-update-offline-trajectory-evaluation --dry-run
python scripts\run_xunce_stage21_5_post_update_offline_trajectory_evaluation.py --config configs\xunce_stage21_5_post_update_offline_trajectory_evaluation_v1.json --output-root D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_5_post_update_offline_trajectory_evaluation_v1 --repo-root .
```

## Non-Goals

- 不启动新的 PPO update。
- 不发布 checkpoint。
- 不替换 default policy。
- 不连接真实 executor。
- 不启动 canary。
- 不把 smoke improvement 当成正式性能结论。
