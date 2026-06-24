# Stage23.1 Slope-Derived Obstacle Source For Endpoint Theta LOS

目标是采用当前项目选择的简化遮挡模型：不做 DEM 高程 ray interpolation，而是把坡度超过最大爬坡角的 DEM cells 物化为 `slope_blocked_cells`，并作为 endpoint `(x,y,theta)` LOS 的障碍代理。

## 关键设计

- `run_quasi_real_map_path_feedback_bridge.py` 导出 `terrain_layers.slope_deg`、`slope_blocked_cells`、`slope_blocked_source_kind=slope_gt_max_traversable_deg`、`slope_blocked_obstacle_source_kind=slope_blocked_as_obstacle_proxy` 和 `max_traversable_slope_deg`。
- 坡度角用物理角度计算：`atan(abs(dz) / (resolution_m * neighbor_distance_factor))`，8 邻域取最大值；默认阈值为 `20.0` 度。
- high-fidelity obstacle source 优先级为 physical obstacle > slope-blocked proxy > blocked/passable-mask proxy > optional no-go proxy。
- Stage23.1 runner 重跑 Stage23.0A/23.0，并按 slope LOS materiality route 到 Stage23.2 或 audit-only。

## 边界

- `slope_blocked_cells` 是 `slope_blocked_as_obstacle_proxy`，不是真实岩石、墙体或裂缝标注。
- 本阶段不启动 PPO、不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。
- 本阶段不做连续 theta、不做沿路径持续观测、不做 DEM 高度视线插值、不做 3D 遮挡、不修改 default A*。

## 验证

```powershell
python -m pytest tests\test_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los.py tests\test_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los.py tests\test_xunce_stage23_0b_map_obstacle_source_export.py tests\test_xunce_high_fidelity_exploration_coverage_comparison.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage23-1

python -m py_compile scripts\run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los.py scripts\run_quasi_real_map_path_feedback_bridge.py scripts\run_xunce_high_fidelity_exploration_coverage_comparison.py

python scripts\run_stage.py --stage xunce-stage23-1-slope-derived-obstacle-source-for-endpoint-theta-los --dry-run
```
