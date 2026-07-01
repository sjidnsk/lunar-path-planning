# Stage26.8H Resumable Diverse Scenario Post-Update Eval

## Summary

目标：把 Stage26.8G 已完成的 diverse scenario collector/update 结果继续推进到可恢复的 Stage26.3 post-update eval。Stage26.8H 不重跑 Stage26.1，也不重跑 Stage26.2；它只把 Stage21.5 的 pre/post high-fidelity eval 拆成可恢复 phase，并在 pre/post 都完成后用 Stage26.3 artifact-only 模式聚合。

## Contracts

- 输入固定为 Stage26.8G root，允许 Stage26.8G 没有最终 summary。
- 必须确认 `h16_seed260801/s26_1` 和 `h16_seed260801/s26_2` 已 passed，且 scenario fixture 至少 3 个 diversity hash 唯一。
- phase 顺序为 `pre_eval -> post_eval -> stage26_3_aggregate`。
- `stage26_3_aggregate` 必须设置 `execute_high_fidelity_evaluations=false`，并显式传入 pre/post artifact roots。
- 主指标仍为 `main_coverage_per_100m_delta`；AUC、总路程、Hybrid A* path cost 只作诊断。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。

## Review Gates

1. Phase contract review：确认 Stage26.8H 不重跑 collector/update，不改 reward、PPO、network、Hybrid A* 或 synthetic terrain。
2. Eval resume review：确认 pre/post roots 可独立完成，已 passed phase 不重跑，aggregate 不会再次启动 high-fidelity eval。
3. Result review：确认 strong join、scenario diversity recheck、safety/fallback/unreachable 和 `main_coverage_per_100m_delta` 可读。

## Validation

```powershell
python -m pytest tests\test_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval.py tests\test_xunce_stage26_8g_repair_synthetic_scenario_diversity.py tests\test_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-8h

python -m py_compile scripts\run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval.py scripts\run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py scripts\run_xunce_stage21_5_post_update_offline_trajectory_evaluation.py

python scripts\run_stage.py --stage xunce-stage26-8h-resumable-diverse-scenario-post-update-eval --dry-run
```
