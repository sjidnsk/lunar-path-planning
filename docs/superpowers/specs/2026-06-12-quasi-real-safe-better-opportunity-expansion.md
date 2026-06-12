# Quasi-Real Safe-Better Opportunity Expansion v1

## 背景

当前准真实 safe-alternative diagnosis 已证明：12 个 LOLA quasi-real context 覆盖 4 个 ROI group，`context_id_missing=0`、exclusion=0、系统 gate regression 全为 0，但 `safe_alternative_context_count=0`、`safe_better_opportunity_context_count=0`。因此 blocker 不是 policy training，而是准真实 ROI/start/target 没有生成可用的安全改选机会。

## Current Evidence Result

本阶段实现后，expansion root `outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/` 已扩展到 108 个 quasi-real context、4 个 ROI group 和 9 个 start cell。bridge、domain-gap、context id、invalid action mask、fallback/open-grid、安全、contract、path/risk 与 source-selection gate 均无回归。

最终 strict `top_k=3` diagnosis 仍显示：

- `safe_alternative_context_count=0`
- `safe_better_opportunity_context_count=0`
- `roi_group_with_safe_better_opportunity_count=0`
- `opportunity_missing_count=108`

因此当前结果是：准真实路面和 path-feedback 链路可用，但当前 ROI/top-k 下没有客观 safe-better 岔路口。

## Role In Roadmap

该阶段现在定位为 **增益支线诊断**，不是准真实 teacher-equivalent 主线 blocker。找不到 safe-better opportunity 阻塞的是“超过或补充 teacher”的 value 分支，不阻塞“policy 学会 teacher/source-selection”的主线。

下一步主线应进入 `Quasi-Real Teacher-Equivalent Validation`：验证 policy 是否能在准真实 ROI 上稳定、安全地复现 source-selected teacher。safe-better 搜索可以继续作为 `Safe-Better Opportunity Search` 并行推进。

## 目标

扩展 LOLA quasi-real ROI/start 支持，生成更多真实地形切片，使 strict `top_k=3` path-feedback 中出现 gate-safe 且 better 的 alternative。该阶段只做机会生成与诊断，不训练、不接管、不写 PPO transition。

## 实现范围

- `RoiSpec` 增加可选 `start_cell`，默认 `[0,0]`，保持旧 manifest 兼容。
- bridge 将 `start_cell` 写入 `LolaSouthPoleRoiConfig`、`current_cell`、slice metadata 与 context id。
- 新增 `configs/quasi_real_safe_better_opportunity_expansion_v1.json`。
- 新增 `scripts/run_quasi_real_safe_better_opportunity_expansion.py/.sh`。
- 新增 `scripts/run_quasi_real_safe_better_opportunity_expansion_closure.sh`。
- 生成 matrix：`model-explorer/data/manifests/lunar_south_pole_lro_lola_safe_better_opportunity_matrix_v1.json`。
- 输出 root：`outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/`。
- readiness 新增 `--quasi-real-safe-better-opportunity-expansion-summary` 与状态 `quasi_real_safe_better_opportunity_expanded`。

## 验收

- expansion matrix 至少覆盖 4 个 ROI group，`candidate_context_count>=48`。
- `start_cell_missing_count=0`，`context_id_missing_count=0`。
- train/validation/test/benchmark split 不复用 `context_id`，`scenario_id_overlap_count=0`。
- final strict `top_k=3` diagnosis passed，`reason_codes=[]`。
- `quasi_real_context_count>=24`，`roi_group_count>=4`。
- invalid action mask、fallback/open-grid、安全、contract、path/risk、source-selection regression 全为 0。
- `safe_alternative_context_count>=8`。
- `safe_better_opportunity_context_count>=4`。
- `roi_group_with_safe_better_opportunity_count>=2`。
- readiness 通过时为 `quasi_real_safe_better_opportunity_expanded` 且 `training_blockers=[]`。
- 若机会不足，必须保留 `next_required_change=quasi_real_roi_start_target_expansion_required`，不得进入训练。

## 验证命令

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  model-explorer/tests/test_quasi_real_data_pipeline.py \
  tests/test_quasi_real_safe_alternative_opportunity_diagnosis.py \
  tests/test_quasi_real_map_domain_gap_evaluation.py \
  tests/test_policy_training_readiness_review.py \
  tests/test_quasi_real_safe_better_opportunity_expansion.py

PYTHON=$P bash scripts/run_quasi_real_safe_better_opportunity_expansion_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --quasi-real-safe-alternative-opportunity-summary \
    outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/quasi-real-safe-alternative-opportunity-summary.json \
  --quasi-real-safe-better-opportunity-expansion-summary \
    outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/quasi-real-safe-better-opportunity-expansion-summary.json \
  --validate-only
```

## 非目标

- 不启动 PPO update。
- 不写 PPO transition。
- 不执行 policy takeover。
- 不发布或替换 checkpoint。
- 不修改 network/action space/default A*。
- 不放宽 distance、path/risk、source-selection gate。
- 不宣称 Ackermann-feasible trajectory。
- 不把 IRIS/GCS/path-planner 诊断当训练放行。
- 不声明策略性能提升。
