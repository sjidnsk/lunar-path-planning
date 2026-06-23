# Stage 20.1 Same-Candidate Oracle Imitation Evidence

## 目标

Stage 20.1 的目标是补足 Stage 20 的同候选集 teacher-label 样本数量。当前 Stage 20 从 Stage 19 preference audit 中只能得到少量严格 same-candidate 样本，因为独立 oracle 轨迹和独立 Xunce 轨迹会快速分叉。Stage 20.1 改为在 Xunce on-policy 状态上额外计算 reward-rerank oracle teacher action。

## 范围

- 修改 high-fidelity coverage comparison runner，支持 `--emit-on-policy-oracle-teacher-labels`。
- 新增 Stage 20.1 runner，聚合 Xunce on-policy teacher labels。
- 扩展 Stage 20 dataset runner，支持合并 Stage20.1 labels。
- 接入 stage registry 和 Stage18/19/20 pipeline。
- 更新 README、AGENTS、系统架构报告和 Xunce 设计 spec。

## 关键语义

- Xunce 仍按自己的 checkpoint 选择 action 并推进轨迹。
- Oracle 只在同一个 `current_cell`、`covered_cells_hash`、`candidate_set_hash` 上给 teacher label。
- 可训练样本必须满足 `baseline_policy=xunce`、`same_candidate_set=true`、`hard_risk_clean_pair=true`、teacher 与 Xunce action 不同、profile hash 一致。
- Stage20.1 不训练、不启动 PPO、不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。

## 主要产物

- `xunce-exploration-coverage-on-policy-oracle-teacher-labels.jsonl`
- `xunce-stage20-1-same-candidate-oracle-imitation-summary.json`
- `xunce-stage20-1-on-policy-teacher-labels.jsonl`
- `xunce-stage20-1-teacher-label-exclusion-report.jsonl`
- `xunce-stage20-1-dataset-stats.json`
- `xunce-stage20-1-next-stage-routing.json`
- `xunce-stage20-1-report.md`
- `xunce-stage20-1-manifest.json`

## 验证

```powershell
python -m pytest tests\test_xunce_stage20_1_same_candidate_oracle_imitation_evidence.py tests\test_xunce_stage20_reward_rerank_oracle_imitation_dataset.py tests\test_xunce_high_fidelity_exploration_coverage_comparison.py tests\test_xunce_stage18_research_evidence_pipeline.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage20-1-same-candidate-oracle-imitation

python -m py_compile scripts\run_xunce_stage20_1_same_candidate_oracle_imitation_evidence.py scripts\run_xunce_stage20_reward_rerank_oracle_imitation_dataset.py scripts\run_xunce_high_fidelity_exploration_coverage_comparison.py scripts\xunce_stage18_pipeline.py scripts\run_xunce_stage18_research_evidence_pipeline.py

python scripts\run_stage.py --stage xunce-stage20-1-same-candidate-oracle-imitation-evidence --dry-run
```
