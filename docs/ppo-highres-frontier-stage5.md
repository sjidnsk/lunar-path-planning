# PPO High-Resolution Frontier Map Exploration Stage 5

Stage 5 实现公平 Baseline Evaluator。它在相同的 16 个固定场景上，以相同环境、候选集、动作执行器、终止规则和指标定义，比较 `random_valid_frontier`、`nearest_frontier`、`max_potential_gain_frontier`、`gain_over_cost_frontier` 与使用冻结 Stage 4 checkpoint 的 `ppo_policy`。

## 公平性边界

- baseline 决策只读取公开 observation，不读取完整地图、隐藏障碍、未观测真值或 evaluator 分母。
- evaluator 仅在动作执行后使用环境真值计算覆盖率、安全性和最终指标。
- PPO 以 `torch.inference_mode()` 运行；不训练、不更新 checkpoint、不替换 default policy。
- 公平性合同共享环境、场景/seed、候选生成、planner、sensor、覆盖分母与预算；
  唯一允许不同的是 method-specific action rule。四个 non-learning baseline 按各自
  固定规则选择 frontier index，并执行该候选的 `recommended_theta`；PPO 对有效
  logits 做最低 index 稳定 argmax，并执行选中候选的 policy `theta_mu`。
- 固定场景和 evaluation seed 对五种方法完全一致，bootstrap 仅对 episode 单元进行固定种子的有放回重采样。
- 空候选集统一由环境终止为 `no_candidate_done`，不允许伪造动作或 logprob，也不允许绕过候选生成或动作安全检查。

本阶段的 claim boundary 为 `fair_baseline_evaluator_system_closure_no_task_advantage/v1`。它只证明公平 evaluator 的系统闭环与可复现机器证据，**不建立 PPO 性能优势**，也不授权 Stage 6、checkpoint 发布、executor、canary、default policy 变更或任何部署动作。

## 机器产物

正式运行写入 `D:/xunce/out/ppo_frontier/<run-id>/s5/`。为便于 fail-closed 校验精确 artifact 集，标准机器文件与 episode traces、公平性审计、bootstrap 审计、冻结 Stage 4 authority 审计、执行身份及比较表均以固定 canonical 文件名平铺在该根目录。完成态只能路由到 `awaiting_independent_review`，实现者不生成 review、approval 或 gate artifact。
