# Agent Notes

本文件只保存长期操作规则和当前主线硬边界；完整计划在 `docs/superpowers/plans/`，真实结果在 D 盘对应 output root 的 `report.md`。

## 沟通与文件安全

- 默认使用中文；包含中文的 `.md`、`.json`、`.py`、`.txt` 必须 UTF-8 编码。修改中文文件优先用 `apply_patch`，并以显式 UTF-8 读取核验。
- 禁止递归或批量删除：不得使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。删除只能是单个明确路径。
- 下载、数据集、模型、缓存、导出和运行时 artifact 默认写入 `D:/CodexDownloads` 或 `D:/xunce/out/<stage_short>`；不提交训练输出、checkpoint 或 job state。
- 不得 `git reset --hard`、`git checkout -- <path>`、批量清理未跟踪文件或覆盖用户改动。提交时只纳入当前任务文件。

## 路径与 artifact 约定

- 新 runner/config/test/plan 分别使用 `scripts/run_xunce_<stage_short>_<purpose>.py`、`configs/xunce_<stage_short>_<purpose>_v1.json`、`tests/test_xunce_<stage_short>_<purpose>.py`、`docs/superpowers/plans/YYYY-MM-DD-xunce-<stage-id>.md`。
- 运行时 artifact 使用短 canonical 名：`summary.json`、`manifest.json`、`routing.json`、`report.md`、`config.json`、`state.jsonl`、`job-state.jsonl`、`phase-state.jsonl`。
- 主线 runner 使用 `scripts/xunce_artifact_io.py` 与 `scripts/xunce_artifact_paths.py`；新 output root 推荐小于 80 字符，完整 artifact path 超过 180 字符为 warning，达到 240 字符必须缩短。

## 当前主线与硬边界

- 仅保留 Stage6 高分辨率前沿 PPO、G1/G2/G3 双门槛链、默认 Python grid A*、opt-in path-planner v3 和 `dev-platform-constraints`。
- 默认路径规划为 `path_planner.search.AStarPlanner`；Hybrid A* 与 v3 均为 opt-in，Hybrid A* 不宣称 Ackermann feasible。
- `max_traversable_slope_deg=30.0` 是平台对齐硬阈值。
- synthetic terrain 只能标为 proxy，不得写成 `physical_obstacle_cells`。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。
- `path-planner` 和 `dev-platform-constraints` 是保留 gitlink。已退役路线仅可从备份或外部 artifacts 恢复，不得重新加入历史 stage runner。
