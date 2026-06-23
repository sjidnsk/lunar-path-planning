# Stage 21.4 Tiny PPO Update Smoke Plan

## Summary

Stage 21.4 的目标是在 Stage21.3 已验证的 PPO batch 上执行一次极小规模 Xunce full-network PPO update smoke，验证 loss、ratio、clip、entropy、value loss、KL、梯度和 experimental checkpoint 写入都可工作。本阶段允许写实验 checkpoint 到 D 盘，但不发布、不替换 default policy、不连接 executor、不启动 canary，也不宣称策略性能提升。

## Inputs

- Stage21.3 PPO batch validation root
- Stage21.1 collector summary，用于读取原始 Xunce checkpoint lineage
- 当前 Xunce candidate checkpoint

## Outputs

- `xunce-stage21-4-tiny-ppo-update-smoke-summary.json`
- `xunce-stage21-4-ppo-loss-audit.jsonl`
- `xunce-stage21-4-gradient-audit.json`
- `xunce-stage21-4-checkpoint-audit.json`
- `xunce-stage21-4-next-stage-routing.json`
- `xunce-stage21-4-report.md`
- `xunce-stage21-4-manifest.json`
- `experimental-xunce-stage21-4-tiny-ppo-candidate.pt`
- `experimental-xunce-stage21-4-tiny-ppo-candidate-metadata.json`

## Implementation Rules

- 只消费 Stage21.3 JSONL 中 `stage21_3_split == "train"` 的行。
- 反序列化每行的 `xunce_batch`，使用 Xunce full network 的 `edge_features`、`edge_index`、`memory_features`、`context_features`、`candidate_features`、`candidate_missing_indicators` 和 `action_mask`。
- 加载原始 Xunce checkpoint，不修改原文件。
- 对同一 `action_index` 重算新 logprob/value，计算 PPO clipped policy loss、value loss、entropy、approx KL、clip fraction。
- 新 logprob 必须使用 Stage21.1 记录的 `sampling_mask` 和 `sampling_temperature`，不能只用 `action_mask`；否则 PPO ratio 与采样分布不一致。
- 默认只跑 `epochs=1`、`learning_rate=1e-5`、`clip_ratio=0.2`，并支持配置。
- 检查 total/policy/value/entropy/KL/grad norm 全部 finite。
- checkpoint 只能写入 D 盘 output root，metadata 必须标记 `experimental_only=true`、`publishes_checkpoint=false`、`replaces_default_policy=false`。
- 若 train batch 太小，本阶段仍可通过 smoke，但必须写 `sample_count_too_low_for_performance_claim=true`。

## Routes

- Missing/stale input: `rerun_stage21_4_required_inputs`
- Boundary flag violation: `resolve_stage21_4_tiny_ppo_boundary_rejections`
- Batch contract failure: `repair_stage21_4_train_split_batch_contract`
- Non-finite loss/gradient/KL: `repair_stage21_4_ppo_update_numerics`
- Checkpoint write/load failure: `repair_stage21_4_experimental_checkpoint_isolation`
- Passed smoke: `implement_stage21_5_single_seed_ppo_pilot`

## Acceptance Criteria

- Stage21.3 summary is passed and routes to Stage21.4.
- Train split row count is greater than zero.
- PPO loss components and approx KL are finite.
- Gradient norm is finite and at least one trainable parameter receives gradient.
- Experimental checkpoint exists, has SHA256/size metadata, and can be loaded into `XunceFullNetworkV1`.
- Source checkpoint remains read-only; output checkpoint path is under D drive Stage21 output root.
- `runs_new_ppo_update=true` is allowed only for this offline smoke, while `publishes_checkpoint=false`, `replaces_default_policy=false`, `connects_real_executor=false`, `starts_online_canary=false`.
- Route is `implement_stage21_5_single_seed_ppo_pilot` only after the smoke update and checkpoint audit pass.
