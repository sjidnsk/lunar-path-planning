# Quasi-Real Teacher-Equivalent Roadmap v1

## 背景证据

最新 quasi-real safe-better expansion root 为 `outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/`。
它覆盖 108 个 LOLA quasi-real context、4 个 ROI group 和 9 个 start cell。bridge、domain-gap、context id、
invalid action mask、fallback/open-grid、安全、contract、path/risk 与 source-selection gate 均无回归。

但 strict `top_k=3` diagnosis 仍为：

- `safe_alternative_context_count=0`
- `safe_better_opportunity_context_count=0`
- `roi_group_with_safe_better_opportunity_count=0`
- `opportunity_missing_count=108`

这说明当前准真实 ROI/top-k 下没有客观可用的 gate-safe + better alternative。该结果不应继续被解释为 policy
没有学会 teacher；在真实/准真实地形中，teacher/source-selection 本来就可能已经是 path/risk/contract 下的最优或近似最优选择。

## 路线重排

后续路线分为主线和增益支线：

- 主线：验证 policy 是否能在准真实地形中稳定、安全地复现 teacher。
- 增益支线：继续寻找 safe-better 或 source-selection blind spot，用于证明 policy 能补充或超过 teacher。

`quasi_real_safe_alternative_opportunity_gap` 只阻塞 safe-better/value 支线，不阻塞 teacher-equivalent 主线。

## 主线阶段

1. `Quasi-Real Teacher-Equivalent Validation`
   - 验证 policy 在 quasi-real ROI 上复现 source-selected action。
   - 高 source-aligned rate 可接受。
   - 失败条件是 unsafe disagreement、gate regression、context/contract/action-mask 断裂。

2. `Quasi-Real Teacher Distillation Robustness`
   - 扩大 ROI、split、start/target 覆盖。
   - 建立 train/validation/holdout 隔离。
   - 验证 teacher agreement 不依赖单一 ROI、seed 或 slice。

3. `Quasi-Real Guarded Teacher-Following Pilot`
   - policy 提出 action。
   - teacher-aligned step 视为有效受控行为。
   - changed choice 必须过 action mask、reachable、fallback、安全、contract、path/risk、source-selection gate。
   - rejected changed choice 回退 teacher 并进入 diagnostic。

4. `Quasi-Real PPO Collector Dry-Run`
   - 只物化 contract-valid 的 teacher-following 或 gate-passed policy transition。
   - source fallback、rejected changed choice、不可 action-bind 样本不得进入 PPO trainable。

5. `Limited Quasi-Real PPO Update Smoke`
   - 从 experimental checkpoint 初始化。
   - 用准真实 PPO-trainable transition 做极小 update。
   - update 后必须保持 generated sequential、quasi-real teacher-equivalent、quasi-real guarded、collector gate 全无回归。

6. `Broader Real/Quasi-Real Domain Evaluation`
   - 扩大 LOLA ROI 区域、地形类型、观测质量和 holdout 覆盖。
   - 验证 teacher imitation 与 guarded behavior 的域稳健性。

## 增益支线

- `Safe-Better Opportunity Search`：继续寻找真实存在的 gate-safe + better candidate，但找不到不阻塞主线。
- `Source-Selection Blind Spot Mining`：当 safe-better 存在而 teacher 未选时，定位 source-selection 盲点。
- `Reward/Preference Calibration`：只在存在明确 preference 或 hard-negative 证据时校准，不把 source-aligned 误判为失败。

## 验收原则

- teacher-equivalent 主线验收 source agreement、gate cleanliness、context id 完整性和无回归。
- safe-better/value 支线验收 accepted better choice 或 source-selection blind spot，但不作为 teacher-equivalent 前置条件。
- 任何阶段都必须保留 git provenance，旧 evidence 不能给新 HEAD 背书。

## 非目标

- 不宣称正式 PPO ready。
- 不发布或替换 checkpoint。
- 不修改 network/action space/default A*。
- 不放宽 distance、path/risk、source-selection contract。
- 不宣称 Ackermann-feasible trajectory。
- 不把 IRIS/GCS/path-planner 诊断当训练放行。
- 不声明策略性能提升。
