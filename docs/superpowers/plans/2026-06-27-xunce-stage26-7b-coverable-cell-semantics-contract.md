# Stage26.7B Coverable Cell Semantics Contract

## Goal

完善 Stage26 synthetic terrain 覆盖率定义，区分四类格子语义：

- `traversable_cells`: 可通行地表。
- `los_blocker_cells`: 会遮挡 endpoint theta LOS 的格子。
- `main_coverable_cells`: 主覆盖率分母，表示安全可探索地表。
- `hazard_observable_cells`: 岩石、深坑、陡坡等危险目标，单独统计侦察率。

本阶段只重算 coverage denominator 和审计字段，不运行 PPO，不修改 reward/network/Hybrid A*/synthetic 生成逻辑。

## Contract

- 新增 `coverage_denominator_mode=main_coverable_cells`。
- `main_coverable_cells = passable_mask=true - physical_obstacle - slope_blocked - blocked - synthetic_hard`。
- `synthetic_los_blocker_cells` 不自动从主覆盖分母扣除，除非它同时属于 hard obstacle 或不可通行。
- `hazard_observable_cells = physical_obstacle + slope_blocked + synthetic_hard + synthetic_high_risk`。
- 输出 `main_coverage_rate`、`raw_roi_coverage_rate`、`passable_coverage_rate`、`hazard_observation_rate` 和 `coverable_cell_semantics_hash`。

## Implementation

- 修改 `scripts/run_xunce_high_fidelity_exploration_coverage_comparison.py`，新增 sidecar-derived semantics helper。
- 新增 `scripts/run_xunce_stage26_7b_coverable_cell_semantics_contract.py` 做只读审计。
- 新增 `configs/xunce_stage26_7b_coverable_cell_semantics_contract_v1.json`。
- 新增 `tests/test_xunce_stage26_7b_coverable_cell_semantics_contract.py`。
- 注册 stage id `xunce-stage26-7b-coverable-cell-semantics-contract`。

## Routing

- 合同完整：`rerun_stage26_7_path_efficiency_with_main_coverable_coverage`
- 输入缺失：`rerun_stage26_7b_required_inputs`
- 语义混淆：`repair_stage26_7b_cell_semantics_contract`
- hard obstacle 进入主分母：`repair_stage26_7b_main_coverable_denominator`
- synthetic 被写成 physical：`repair_stage26_7b_synthetic_source_semantics`
- hash 不稳定：`repair_stage26_7b_denominator_hash_contract`

## Verification

```powershell
python -m pytest tests\test_xunce_stage26_7b_coverable_cell_semantics_contract.py tests\test_xunce_high_fidelity_exploration_coverage_comparison.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-7b
python -m py_compile scripts\run_xunce_stage26_7b_coverable_cell_semantics_contract.py scripts\run_xunce_high_fidelity_exploration_coverage_comparison.py
python scripts\run_stage.py --stage xunce-stage26-7b-coverable-cell-semantics-contract --dry-run
```
