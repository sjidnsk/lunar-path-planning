# Agent Notes

## Stage 21.12 Coverage-Constrained Reward Profile Repair

- Stage21.12 的职责是修复 Stage21.11 发现的 reward 与 coverage-per-cost 不对齐问题；目标不是把路径成本变成主目标，而是在覆盖进展相近时偏向更低路径成本。
- 新 profile 是 `configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json`，schema 为 `xunce-stage21-coverage-constrained-ppo-reward-profile/v2`；v1 保留为历史只读兼容，不覆盖。
- v2 reward 继续以 99% final coverage 为第一目标，hard risk 必须 reject-before-reward；`coverage_per_cost_component` 只在 hard-risk clean 且 coverage gain > 0 时生效。
- Stage21.12 的真实输出根是 `D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_12_coverage_constrained_reward_profile_repair_v1`，当前 `status=passed`，下一跳是 `run_stage21_13_coverage_constrained_multi_seed_ppo_smoke`。
- 当前证据：v1 reward best matches coverage-per-cost rate 为 `0.20833333333333334`，v2 为 `0.29583333333333334`，v2 alignment improvement 为 `0.0875`，v2 coverage-priority violation rate 为 `0.020833333333333332`，v2 advantage 与 coverage-per-cost 相关为 `0.318183174323286`，hard-risk trainable count 为 `0`。
- Stage21.12 只生成 recommended Stage21.2/Stage21.6 config，`runs_new_ppo_update=false`；不启动正式 PPO，不发布 checkpoint，不替换 default policy，不连接 executor，不启动 canary，不修改 network/action space/default A*。

## Stage 21.11 Coverage-Constrained Path-Cost Objective Audit

- Stage21.11 的职责是解释 Stage21.10 中“PPO update 已稳定但 final coverage/AUC 仍无提升”的原因。
- 审计顺序是：Stage21.2 reward 是否支持 coverage-per-cost、Stage21.3 advantage 是否给高 coverage-per-cost 动作正信号、Stage21.4/21.5 pre/post probability 是否真的改变、Stage21.5 evaluation binding 是否可信。
- 绑定 pre/post action probability 时必须优先使用强状态键：`scenario_id + step_index + candidate_set_hash + covered_cells_hash + current_cell`；缺少强绑定字段时只能写 diagnostic，不能把弱 join 当成最终行为变化证据。
- Stage21.11 的目标描述是“以尽量低路径成本达到 99% 覆盖”，不是把路径成本降到最低而牺牲覆盖。低覆盖但低成本不能算任务成功。
- Stage21.11 只做离线审计和推荐配置，不启动 PPO、不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。

## Stage 21.7 Reward / Collector / Advantage / Horizon Repair

- Stage 21.7 的职责是解释并修复 Stage21.6 “3 个 seed 都完成离线 PPO update，但 final coverage/AUC 没提升”的原因。
- 主要审计五类信号：reward 是否弱或错向、advantage 是否平或错向、trainable transition 是否不足、PPO update 是否太弱、holdout horizon 是否太短。
- 当前 Stage21.6 只有 24 条 trainable transition，低于 200 条性能声明门槛；4-step holdout 也不能证明 40-step/99% 覆盖能力。
- Stage21.7 可以生成 repaired Stage21.6 smoke config，并执行一次本地 repaired pilot smoke；该 smoke 仍不是正式 PPO 训练或发布授权。
- Stage21.7 必须保持 `publishes_checkpoint=false`、`replaces_default_policy=false`、`connects_real_executor=false`、`starts_online_canary=false`、`canary_traffic_fraction=0.0`。
- Stage21.7 不修改 network/action space/default A*，不把 smoke 结果宣称为真实性能提升。

## Stage 21.8 PPO Update Strength Calibration

- Stage 21.8 的职责是校准 Stage21.7 repaired pilot 暴露出的 PPO 更新强度问题，尤其是 `grad_unstable`。
- Stage21.4 的 `max_grad_norm` 是训练时的梯度裁剪阈值；Stage21.6/21.8 的 pre-clip grad gate 是审计门槛，二者必须分开记录。
- Runtime-blocked combo 默认不自动重试；如果要重试，必须显式设置 `retry_runtime_blocked_combinations=true`，或换新的 `combo_id` / `sweep_work_root`。
- 缺少 seed 阶段状态、batch fingerprint、transition fingerprint、post-clip grad norm 时，不能推荐配置。
- Stage21.8 输出的 recommended Stage21.6 config 只允许来自 Stage21.6 `status=passed` 的 sweep，并且只能用于下一次 repaired multi-seed pilot。
- Stage21.8 必须保持 `publishes_checkpoint=false`、`replaces_default_policy=false`、`connects_real_executor=false`、`starts_online_canary=false`、`canary_traffic_fraction=0.0`，且不修改 network/action space/default A*。

## Stage 21.9 Gradient Normalization / Loss Scaling Repair

- Stage 21.9 的职责是修复 Stage21.8 证明的 PPO 原始梯度过大问题；降低 learning rate 只缩小参数步长，不能明显降低 `pre_clip_grad_norm`。
- Stage21.9 优先审计 Stage21.3 train split advantage normalization 是否生效，并定位 Stage21.4 的 policy/value/entropy/total loss 哪一项主导梯度。
- Stage21.9 默认只修 loss/advantage scale：`advantage_clip_abs=5.0`、`normalize_minibatch_advantages=true`、`loss_scale=0.25`、`value_loss_coefficient=0.1`；不改 reward 目标、collector、network、action space 或 default A*。
- Stage21.9 repaired smoke 必须直接读取 Stage21.8 sweep row 的 `pre_clip_grad_norm_max`、post-clip norm、KL、entropy、parameter delta、policy shift 和 coverage/AUC 字段，不能把 Stage21.4 `status=passed` 误当作梯度稳定。
- 通过 Stage21.9 只允许生成可再次尝试 Stage21.6 的 repaired config；仍不启动正式 PPO 训练、不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。

## Stage 21.10 Stage21.9 Repaired Multi-Seed PPO Pilot

- Stage21.10 的职责是用 Stage21.9 修复后的 loss/advantage scale 重新执行 Stage21.6 multi-seed pilot，验证梯度稳定后 coverage/AUC 是否开始提升。
- Stage21.10 必须生成专用 Stage21.4 config：`learning_rate=2e-6`、`epochs=1`、`clip_ratio=0.2`、`max_grad_norm=1.0`，并保留 `advantage_clip_abs=5.0`、`normalize_minibatch_advantages=true`、`loss_scale=0.25`、`value_loss_coefficient=0.1`。
- Stage21.4 的 `max_grad_norm=1.0` 是训练时 post-clip 限制；Stage21.6/21.10 的 `stage21_6_pre_clip_grad_norm_gate=25.0` 是审计门槛，二者必须分开记录。
- Stage21.10 不能只看 Stage21.6 的 raw `mean_final_coverage_delta` / `mean_coverage_auc_delta`；还必须读取每个 seed 的 Stage21.5 capped final coverage/AUC delta。只有 raw 与 capped mean 都严格大于 0，且 worst seed 不回退，才允许 route 到 `scale_stage21_ppo_pilot_scenarios_and_horizon`。
- Windows 下 Stage21.10 默认使用 `stage21_6_output_subdir=s6` 作为内部 Stage21.6 执行目录，以避免长输出根叠加 high-fidelity artifact 名称后超过 260 字符路径限制。
- 当前 Stage21.10 真实 run 已完成 3 seeds / 240 transitions，lineage 与边界通过，`pre_clip_grad_norm_max=2.334965467453003`，但 raw/capped final coverage 与 AUC delta 全为 0；下一跳是 `repair_stage21_reward_signal_or_advantage_separation`。
- Stage21.10 仍是离线 pilot，不是正式 PPO 训练或发布授权；必须保持 `publishes_checkpoint=false`、`replaces_default_policy=false`、`connects_real_executor=false`、`starts_online_canary=false`、`canary_traffic_fraction=0.0`。
## Stage 21 Pure PPO 主线

- Stage 21 的新主线是纯 PPO：`Xunce on-policy rollout -> PPO trainable batch -> coverage-first reward -> tiny PPO update -> post-update trajectory evaluation -> multi-seed PPO pilot`。
- Stage 21.0 只做只读 readiness audit，产出 summary/capability audit/routing/report/manifest，不启动 PPO、不写可发布 checkpoint、不替换 default policy、不连接真实 executor、不启动 canary。
- Stage 20/20.1 的 reward-rerank oracle imitation 结果只保留为诊断上下文，不作为 Stage 21 纯 PPO 主线的训练或 readiness 门槛。
- Stage 21 的大体量输出默认写入 `D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\`。
- 推进 Stage 21.1 之前必须先通过 `xunce-stage21-0-pure-ppo-readiness-audit`，并确认下一跳为 `implement_stage21_1_xunce_on_policy_ppo_rollout_collector`。
- Stage 21.1 是 Xunce on-policy PPO rollout collector：必须用 stochastic sampling 采样动作，记录 `old_log_prob`、`old_value`、`xunce_batch`、reward、next observation 和 done，不得复用 argmax evaluator 作为 PPO 样本来源。
- Stage 21.1 的 trainable batch 必须使用 `sampling_mask = action_mask & hard_risk_clean_mask`。reachable 但含 hard risk flag 的候选只能进入审计/拒绝报告，不能进入 PPO trainable transition。
- Stage 21.1 仍然不训练、不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary；通过后只允许进入 Stage 21.2 coverage-first reward contract。
- Stage 21.2 新增 Stage21 专用 coverage-first PPO reward profile/helper，不覆盖 canonical v3。99% final coverage 是第一成功 gate；path cost 和 soft risk 只是约束与次级优化。
- Stage 21.2 的 hard risk 必须 reject-before-reward，不能被覆盖收益抵消；当 path cost 已包含 risk proxy 时，soft risk 只能作为 tiny/audit-weighted penalty，避免重复扣风险。
- Stage 21.2 仍然不训练、不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary；通过后只允许进入 Stage 21.3 PPO batch validation。

## Stage 20 Reward-Rerank Oracle Imitation 边界

- Stage 20 的职责是把 Stage 19 的 reward-rerank oracle preference audit 转成严格同候选集 teacher-label 数据集，并做小规模 teacher-imitation dry-run。
- 只有 `baseline_policy=xunce`、`same_candidate_set=true`、`hard_risk_clean_pair=true` 且 oracle 与 Xunce 选择不同 action 的行可以进入 trainable samples；跨轨迹、不同 candidate set、incumbent-only 或 hard risk 不干净的行只能进入 exclusion report。
- 当前严格样本预计只有约 24 条，dry-run 成功也只能说明数据管线和 imitation loss 可执行，不能说明 Xunce 已经学会 oracle。
- 少于 200 条严格 same-candidate teacher samples 时，下一跳应是 `collect_more_reward_rerank_same_candidate_preference_evidence`，而不是 checkpoint 训练或 PPO。
- Stage 20 不启动 PPO、不发布 checkpoint、不替换 default policy、不连接真实 executor、不启动 canary；`stage20_authorized` 与 `training_or_release_authorized` 必须保持 `false`。

## Stage 20.1 Same-Candidate Oracle Imitation Evidence 边界

- Stage 20.1 的职责是补采 Xunce on-policy 状态上的 reward-rerank oracle teacher label。Xunce 仍按自己的 checkpoint 选择和移动，oracle 只在同一个当前状态、同一个候选集上计算 teacher action。
- Stage 20.1 产物可以补充 Stage 20 数据集，但不训练模型、不启动 PPO、不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。
- 可训练 label 必须满足：`baseline_policy=xunce`、`same_candidate_set=true`、`hard_risk_clean_pair=true`、`teacher_action_index != xunce_action_index`，且 `teacher_profile_hash` 与 Stage 19 practical target 一致。
- Stage 20 合并 Stage20.1 样本后，只有达到 200 条以上并通过 dry-run，才允许路由到 `stage20_1_supervised_oracle_imitation_checkpoint_preflight`；该 route 仍不是训练或发布授权。

## 沟通语言

- 默认使用中文回答，除非用户明确要求使用其他语言。

## Notion 导出规则

- `D:\codex\project\lunar-path-planning\docs\月面巡视探索.md` 是 Notion 页面“月面巡视探索”的单向导出副本。
- Notion 为主源，本地 Markdown 文件为副本。
- Notion 内容更新后，需要重新导出覆盖本地文件。
- 本地文件应保持 UTF-8 编码。

## 项目目标优先级

- 探索任务的最高目标是整场任务最终覆盖率超过 99%。当覆盖率尚未达到 99% 时，后续方案、reward 调整、审计和阶段推进应优先解释并解决覆盖不足问题。
- 路径成本、soft risk exposure、coverage efficiency 是重要约束和次级优化目标，但不能替代 99% 覆盖率这个主指标。不要因为路径更短或风险代理更低，就把低覆盖方案描述为任务成功。
- 风险系统的职责是过滤不允许走的路径；reward 的职责是在允许路径中选择更高覆盖、更低成本的行动。hard risk violation 必须保持为 0；soft risk 只应在可接受范围内约束探索，不应把策略压成过度保守。
- 路径成本优化的目标是在保持或推进 99% 覆盖率的前提下降低绕路。若出现“覆盖率明显不足但路径成本很好”的结果，应优先视为覆盖目标未完成，而不是成功收敛。
- 阶段报告和下一阶段计划必须明确区分：最终覆盖率、覆盖增量、路径成本、单位路径覆盖效率、hard risk violation 和 soft risk exposure。结论排序应以“是否接近或达到 99% 覆盖率”为第一判断，再讨论成本和风险是否可接受。

## Goal 模式规则

- Goal 模式的 prompt 不能超过 4000 字符。
- 当用户要求“下一阶段 goal 模式 prompt”或“goal 模式完整 prompt”时，输出应是可直接粘贴执行的 `/goal` 文本，而不是泛泛路线图。
- Goal prompt 必须基于当前仓库状态、最新 evidence/root、readiness blocker 和已实现/未完成边界来写；不要脱离当前证据重新发散方案。
- Goal prompt 默认包含：背景、目标、范围、验收标准、验证命令、产物路径、非目标。
- 每次编写下一阶段 Goal prompt 时，必须把“更新项目文档”写入范围和验收标准；已知文件时明确列出，例如 `docs/算法设计与系统架构报告.md` 与 `docs/superpowers/specs/`。
- Goal prompt 应保留关键 scope guards，例如不启动 PPO、不修改 network/action space/default A*、不宣称 Ackermann-feasible trajectory、不把 IRIS/GCS 诊断当训练放行。
- 若用户明确要求 4000 字符以内，应从一开始按该预算压缩，优先保留验收门禁、关键 artifact、验证命令和非目标，删减重复解释。

## 开发环境

- 项目开发和验证默认使用 Conda 环境 `lunar-explorer`：`/home/kai/anaconda3/envs/lunar-explorer`。

## Stage 19 / Stage 20 证据边界

- Stage 19 的职责是 evaluator / critic preflight：把 Stage 18.11 的 reward-rerank oracle 诊断结果整理成人工审查证据，不训练模型。
- 当前 practical target 为 `candidate_count=36`、`path_cost_weight=0.1`；判断 99% 覆盖目标时必须使用 capped final coverage，raw coverage 超过 1.0 只能作为饱和诊断。
- 固定 Xunce checkpoint 是否真正学会 oracle 行为必须单独说明；当 `xunce_checkpoint_advantage_established=false` 时，不得把 oracle 结果写成 Xunce 模型优势。
- Stage 19 通过后的下一跳可以是 `stage20_reward_rerank_oracle_preference_dataset_preparation`，但 `stage20_authorized=false`。Stage 20 仍是偏好证据准备和人工审查，不是 PPO 训练、checkpoint 发布、default policy 替换、executor 连接或 canary 启动。

# Stage 21.3 PPO Batch Validation 边界

- Stage 21.3 是 PPO update 前的训练输入门禁，只验证 batch，不训练、不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。
- Stage 21.3 必须把 Stage21.1 transition 与 Stage21.2 reward evaluation 按 `transition_id` 一对一 join；空 ID、重复 ID、missing/extra reward row 都必须阻止进入 Stage21.4。
- Episode contract 必须显式验证：每个 scenario 内 `step_index` 唯一、单调、连续，且恰好一个 terminal transition；terminal 后不得还有 transition。
- Reward trainability 必须强绑定：Stage21.2 reward row 的 `trainable=false` 或 `hard_risk_rejected` 不能进入 PPO trainable batch。
- `action_index` 必须非负，并同时满足 `action_mask`、`sampling_mask`、`hard_risk_clean_mask`；Python 负索引不能被视为有效 action。
- Advantage normalization 只能用 train split 统计量；每条 batch row 必须写入 `stage21_3_split`，后续 Stage21.4 只能消费 train split。
- 当前 Stage21.3 smoke output 位于 `D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_3_ppo_batch_validation_v1`，通过后 route 为 `implement_stage21_4_tiny_ppo_update_smoke`。

# Stage 21.4 Tiny PPO Update Smoke 边界

- Stage 21.4 是纯 PPO 主线中第一个允许离线参数更新的 smoke 阶段；`runs_new_ppo_update=true` 只表示执行了本地 tiny PPO update，不表示训练发布授权。
- Stage 21.4 只能消费 Stage21.3 batch 中 `stage21_3_split == "train"` 的行，不能混入 validation 行。
- Xunce full network 不能直接使用通用 `compute_masked_ppo_loss`；必须用 `XunceFullNetworkV1` 的 candidate/edge/memory/context 输入重新前向，并使用 Stage21.1 记录的 `sampling_mask` 与 `sampling_temperature` 计算 PPO ratio。
- 输出 checkpoint 必须标记 `experimental_only=true`，并写入完整模型维度 metadata，确保可被现有 `_load_xunce_checkpoint` 重载。
- 即使 Stage21.4 通过，也必须保持 `publishes_checkpoint=false`、`replaces_default_policy=false`、`connects_real_executor=false`、`starts_online_canary=false`、`canary_traffic_fraction=0.0`。
- Stage21.4 只验证 loss、KL、grad、parameter_delta 和 experimental checkpoint 保存/重载链路，不宣称覆盖率提升；覆盖效果必须等 Stage21.5 离线 trajectory evaluation 判断。

# Stage 21.5 Post-Update Offline Trajectory Evaluation 边界

- Stage 21.5 只做离线 trajectory 评估，不训练、不再执行 PPO update、不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。
- Stage 21.5 必须同时评估 Stage21.4 的 source checkpoint 与 experimental-only checkpoint，且两次 high-fidelity run 除 checkpoint、output/work-root 外应保持同一 smoke 配置。
- pre/post delta 必须从 `xunce-exploration-coverage-episodes.jsonl` 中按 `policy == "xunce"` 过滤并按 `scenario_id` 一对一比较，不能用 Xunce-vs-incumbent summary delta 代替。
- final coverage、capped final coverage、coverage AUC、capped coverage AUC 任一退化都不能进入 Stage21.6。
- post run 中 hard risk、model inference failure、mask violation、unreachable selected candidate、path-planning failure、open-grid fallback 任一非零都必须阻止进入 Stage21.6。
- Stage21.5 必须继承 Stage21.4 的 `sample_count_too_low_for_performance_claim=true` 边界；通过 Stage21.5 只表示 tiny update 未破坏离线 smoke，不表示模型可发布。

# Stage 21.6 Multi-Seed PPO Pilot 边界

- Stage21.6 是离线多 seed PPO pilot，默认 3 seeds：`2101/2102/2103`，仍使用 2 scenarios x 4 steps 的 conservative smoke。
- 每个 seed 必须重新执行 Stage21.1 -> Stage21.2 -> Stage21.3 -> Stage21.4 -> Stage21.5，使用独立 `sampling_seed`，不能复用同一个 PPO batch 假装多 seed。
- Stage21.6 必须输出并校验 per-seed Stage21.3 batch fingerprint 和 transition-id fingerprint；重复时 route 到 `repair_stage21_6_seed_lineage_or_batch_reuse`。
- Stage21.6 必须把 KL、entropy、grad norm 稳定性纳入 routing gate；数值异常 route 到 `repair_stage21_6_ppo_numerical_stability`。
- 平均 final coverage delta 和 coverage AUC delta 必须严格大于 0，worst seed 不能退化；持平不算成功。
- 低样本 smoke 必须写 `sample_count_too_low_for_performance_claim=true`，不能把结果当正式性能结论。
- Stage21.6 可以设置 `runs_new_ppo_update=true` 表示本地离线 pilot 已运行，但必须保持 `publishes_checkpoint=false`、`replaces_default_policy=false`、`connects_real_executor=false`、`starts_online_canary=false`、`canary_traffic_fraction=0.0`。
