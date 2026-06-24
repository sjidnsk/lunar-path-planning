# Stage23.2A High-Resolution Terrain Data Ingestion

## 目标

Stage23.2A 将 Stage23 的默认样本地图从 20m/cell LOLA quasi-real 数据升级到更高分辨率的月面地形数据，同时继续采用当前简化规则：

```text
slope_deg > max_traversable_slope_deg
=> slope_blocked_cells
=> 不可通行 + endpoint (x,y,theta) LOS 遮挡
```

本阶段只做数据下载、manifest、GeoTIFF 读取、ROI 裁剪和 sidecar 生成，不启动 PPO，不发布 checkpoint，不替换 default policy。

## 数据源

默认主线数据源：

- `lunar_south_pole_usgs_lro_dem_slope_4m`
- USGS Moon LRO South Pole DEM + Slope Map
- 4m/pixel
- manifest: `model-explorer/data/manifests/lunar_south_pole_usgs_lro_dem_slope_4m.json`

ROI 增强候选：

- `lunar_lroc_nac_dtm_roi_2m_5m`
- LROC NAC DTM 2-5m/pixel
- manifest: `model-explorer/data/manifests/lunar_lroc_nac_dtm_roi_2m_5m.json`
- 只有明确选中与当前 ROI 重叠的 DTM 产品后才参与替换或增强。

## 实现范围

- 新增 GeoTIFF window reader：`model-explorer/src/model_explorer/data/geotiff.py`。
- 新增 runner：`scripts/run_xunce_stage23_2a_high_resolution_terrain_data_prepare.py`。
- 新增 config：`configs/xunce_stage23_2a_high_resolution_terrain_data_ingestion_v1.json`。
- 注册 stage id：`xunce-stage23-2a-high-resolution-terrain-data-ingestion`。
- 扩展 `run_quasi_real_map_path_feedback_bridge.py::_sidecar_from_roi`，允许外部 slope map 优先写入 `terrain_layers.slope_deg`。

## 输出

默认输出根：

```text
D:\CodexDownloads\lunar-path-planning\stage23_high_resolution_terrain_ingestion\outputs\path_feedback_batch_xunce_stage23_2a_high_resolution_terrain_data_ingestion_v1
```

主要产物：

- `xunce-stage23-2a-summary.json`
- `xunce-stage23-2a-download-manifest.json`
- `xunce-stage23-2a-dataset-validation-summary.json`
- `xunce-stage23-2a-hash-audit.json`
- `xunce-stage23-2a-roi-overlap-audit.json`
- `high_res_roi_expansion/xunce-high-fidelity-real-map-roi-expansion-summary.json`
- `high_res_roi_expansion/xunce-high-fidelity-real-map-slices.jsonl`
- `high_res_roi_expansion/xunce-high-res-terrain-slices.jsonl`

## 语义边界

- `slope_blocked_cells` 是 `slope_blocked_as_obstacle_proxy`，不是真实岩石、墙体或裂缝标注。
- slope map 优先于 DEM 派生 slope；没有 slope map 时才用 DEM 物理坡度公式。
- raw GeoTIFF 和大文件只写入 D 盘，不提交 Git。
- 旧 20m LOLA 数据保留为 fallback/reference，不删除。
- 本阶段不做连续 theta、不做沿路径持续观测、不做 3D LOS、不修改 default A*。

## 验证

```powershell
python -m pytest tests\test_xunce_stage23_2a_high_resolution_terrain_data_ingestion.py tests\test_xunce_high_fidelity_exploration_coverage_comparison.py tests\test_platform_stage_runner.py -q
python -m py_compile scripts\run_xunce_stage23_2a_high_resolution_terrain_data_prepare.py scripts\run_quasi_real_map_path_feedback_bridge.py
python scripts\run_stage.py --stage xunce-stage23-2a-high-resolution-terrain-data-ingestion --dry-run
```
