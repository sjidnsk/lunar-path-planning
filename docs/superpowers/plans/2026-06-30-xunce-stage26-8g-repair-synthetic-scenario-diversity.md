# Stage26.8G Repair Synthetic Scenario Diversity

## Summary

目标：修复 Stage26.8F 发现的 scenario 内容重复问题。当前已完成 eval 的多个 `scenario_id` 实际从同一 sidecar、同一起点和同一候选序列出发，导致 `main_coverage_per_100m_delta=0` 的解释失真。Stage26.8G 只修 scenario fixture 和审计绑定，不改 reward、PPO、network、Hybrid A* 搜索语义或 synthetic terrain 生成逻辑。

## Key Contracts

- `scenario_diversity_contract_enabled=true`
- `scenario_diversity_source=synthetic_roi_start_seed_matrix/v1`
- 每个 scenario 必须写入 `scenario_seed`、`scenario_start_cell`、`scenario_start_cell_source`、`scenario_roi_id`、`scenario_candidate_seed`、`scenario_diversity_signature_hash`、`scenario_diversity_content_hash`。
- 起点必须从安全可探索格子中确定性选择，排除 physical/slope/blocked/synthetic hard obstacle，并满足最小间距。
- 同一个 synthetic terrain hash 可复用，但不得只靠不同 `scenario_id` 伪装成不同场景。
- `scenario_candidate_seed` 当前只作为可审计 provenance；本阶段不把它强接进确定性的 frontier/NBV 候选生成逻辑，真实多样性由 start/ROI/context 与最终 candidate/action signature 审计确认。

## Execution

1. 读取 Stage26.8F root，确认 `next_required_change=repair_stage26_synthetic_scenario_diversity`。
2. 用 repaired Stage26.1 config 运行 bounded H16 seed260801 smoke，`required_scenario_count=3`。
3. 若 Stage26.1 fixture 多样且 collector passed，再运行 Stage26.2 -> Stage26.3。
4. 对 repaired Stage26.3 输出重新计算 scenario signature duplicate audit。
5. 若多样性修复后仍无动作变化，route 到 policy signal；若出现正向效率信号，恢复 Stage26.8D 队列。

## Review Gates

- Scenario fixture contract review：确认多样性来自真实起点/seed/ROI/candidate seed 差异，synthetic 仍为 proxy。
- Collector/eval binding review：确认 Stage21.1 transition、Stage21.5 inference/step/episode 都携带同一 scenario provenance。
- Full-result review：确认 repaired smoke 不再重复 scenario signature，且下一跳准确。

## Validation

```powershell
python -m pytest tests\test_xunce_stage26_8g_repair_synthetic_scenario_diversity.py tests\test_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit.py tests\test_xunce_stage26_1_synthetic_terrain_collector_smoke.py tests\test_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-8g
python -m py_compile scripts\run_xunce_stage26_8g_repair_synthetic_scenario_diversity.py scripts\run_xunce_stage26_1_synthetic_terrain_collector_smoke.py scripts\run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py
python scripts\run_stage.py --stage xunce-stage26-8g-repair-synthetic-scenario-diversity --dry-run
```
