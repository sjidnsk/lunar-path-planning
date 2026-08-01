# Task 4：移除 Xunce Stage18--25 演进链

## 范围与冻结输入

- 起点提交：`93889ef8b3b98c0a41c1b54ae3222ac61ae38526`。
- 使用仓库内冻结 fixture：`tests/fixtures/route_retirement_candidates_v1.json`。
- fixture 记录的源 manifest SHA-256 为
  `da9f23c3af83c3506143338a41e1c1a8f8f09dc4f59d4a4bac5adea160f6371c`。
- 精确删除 `groups.stage18_25.paths` 的 250 个已跟踪路径：64 个
  config、59 个 script、59 个 test、68 个 doc；未依赖 D 盘 manifest。

## 变更

- 逐个明确路径移除了 fixture 的全部 250 项。
- 从 `configs/stage_registry.json` 局部移除了 57 条仅绑定该链的
  Stage18--25 / `xunce-stage18-research-evidence-pipeline` 入口，保留
  其余 entry 的原有次序和内容；剩余 43 条 entry 的 script 均存在。
- 从 `tests/test_platform_stage_runner.py` 的显式集合移除了同批 57 个
  已删除 stage key，未改变其它 key。
- 删除了 brief 指定的历史 topology 规格
  `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`。
- Stage23 runner 与测试随精确集合删除后，`xunce_terrain_sidecar.py` 不再有
  任何保留路线调用，故作为 orphan closure 删除。
- 新增 `tests/test_retired_stage18_25_cleanup.py`：读取 tracked fixture，断言
  源 manifest SHA、250 及 64/59/59/68 分类、路径在 worktree 与有效 tracked
  集合中均缺失，registry 无 Stage18--25 entry，且所有保留 registry script
  存在；同时检查平台/CI/runner 保留入口没有已删 runner 引用。

## 验证

所有 pytest 均使用 `D:/conda_envs/lunar-explorer/python.exe`，并将当前
`path-planner/src`、`src` 依次置于 `PYTHONPATH` 首位；basetemp 均在
`D:/xunce/basetemp/retired-route-task4-*`。

- `test_retired_stage18_25_cleanup.py`、platform stage runner、retirement
  preflight：38 passed。
- G1/G2/G3 轻量合同及 platform smoke/matrix：234 passed、2 skipped。
- Stage6 标准配置、G1 safe-start fallback、artifact IO/path 在同一次聚焦运行
  中通过（218 passed、3 skipped）；其中完整 `test_stage6_workflow.py` 的 19 项
  会拒绝非空真实 Git index。该 index 正是本任务 staged 删除造成，非 Stage6
  源码或路由回归。尝试以 D 盘隔离的干净 index 重跑完整 workflow 后超过 124 秒
  测试上限，因此没有把完整 workflow 记为通过。
- `run_platform_validation_matrix.py --profile windows-non-drake --dry-run`
  通过；仅生成 dry-run 命令，没有 bootstrap、训练、完整 G 实验、checkpoint、
  executor 或 canary。
- 保留入口静态审计、fixture 缺失审计、registry script 存在性审计均无 offender；
  `git diff --check` 在提交前执行。

## 边界与残余

- 未改动 `src/**`、Stage6 高分辨率 PPO、默认 grid A*、G1/G2/G3、独立 G2
  producer、path-planner v3、dev-platform-constraints、artifact helper 或 30 度
  坡度合同。
- `docs/算法设计与系统架构报告.md` 的纯历史文本入口依 brief 留给 Task5；它不构成
  registry、CI、platform 或保留运行入口的断链。

## Review fix round 1

- 复核发现 `xunce-high-fidelity-exploration-coverage-comparison` 虽未在冻结
  Stage18--25 路径集合内，但其 registry runner 直接导入了已删除的
  `xunce_stage18_guard_thresholds`，属于真实运行时断链和退役早期策略入口。
- 定向删除该 registry entry、runner、config、直属测试、platform runner 的显式
  key 与对应 dry-run 测试；同时删除直接导入该旧 runner 且另含 Stage18I 导入的
  `tests/test_xunce_gate_simplification.py`。
- `tests/test_retired_path_feedback_cleanup.py` 曾手工把已删除的 Stage23 runner
  标为 retained executable；移除了该陈旧特例，保留 registry script 存在性及删除
  runtime-module 扫描合同。
- 两个保留 config 仅以历史 output-root 字符串描述下游输入，未导入或调用上述已删
  文件；按本轮可执行引用边界不修改，留给 Task5 的历史表面收敛。
