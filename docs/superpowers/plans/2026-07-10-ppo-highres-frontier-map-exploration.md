# PPO 高分辨率 Frontier Map 探索 v1 分阶段实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to execute this plan one task at a time. Each task uses TDD, a fresh implementer, and a fresh reviewer. The user-approved gate process below overrides any skill default that would commit before human approval.

**Goal:** 在隔离的 `codex/ppo-highres-frontier-map-exploration` 分支实现一套可训练、可评估、可审计、可复现的 PPO 高分辨率 Frontier Map 探索系统。

**Architecture:** 以 `docs/superpowers/specs/2026-07-09-ppo-highres-frontier-map-exploration-design.md` 为唯一算法基线，新增独立 `src/lunar_exploration_ppo/` 包；环境维护 0.5m highres 状态，策略只消费分层稀疏观测，planner 仅通过 adapter 接入。Foundation Gate 冻结构建、配置、artifact、gate 与运行环境合同，随后 Stage 1–8 串行推进。

**Tech Stack:** Python 3.12、PyTorch 2.12.1+cu130、NumPy、Pydantic、Rasterio、pytest、GitHub Actions、Conda、Windows/Ubuntu CPU CI、NVIDIA CUDA 13.0 runtime。

## 执行与审批合同

- 当前 linked worktree：`C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning`。
- 当前分支：`codex/ppo-highres-frontier-map-exploration`。
- 每个 Stage 内使用 `superpowers:subagent-driven-development`：fresh implementer 按 TDD 实现，fresh reviewer 分别给出规格符合性与代码质量结论。
- Critical/Important 问题必须修复并重审；实施者不得自行提交、暂存或推送。
- 每个 Stage 在机器验收和独立审查通过后停在 `awaiting_human_approval`。只有用户明确批准，主 agent 才创建该 Stage 唯一提交与 `approval.json`、`gate.json`，然后进入下一 Stage。
- 每个获批 Stage 只生成一个 Git 提交；不推送、不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。
- 系统验收与算法性能分开判定。`highres_observed_coverage_rate >= 0.99`；主指标为固定步数预算下的成功率。PPO 未击败 `gain_over_cost_frontier` 不阻止系统验收，但报告必须写明“未建立性能优势”。
- 大型输出、训练产物、checkpoint、环境锁与数据缓存只写 D 盘，不提交 Git。默认根目录：`D:/xunce/out/ppo_frontier/`。

## 固定公共合同

### 环境、数据与安全

- Smoke 使用纯程序化 fixture，不宣称原生 2m 月面数据。
- Standard 使用已有 USGS LRO 4m DEM 的空间不重叠窗口作为真实低分辨率 prior；0.5m highres truth 为确定性程序化 proxy。
- Kilometer 将同类真实 prior 聚合到 8m；0.5m highres 仍为 proxy，仅作压力测试。
- 所有 proxy 固定标记 `synthetic_terrain_obstacle_proxy/v1`、`physical_obstacle_cells_written=false`。
- `value_prior_source=constant_neutral/v1`，禁止从隐藏 highres truth 反推。
- 安全常量：`vehicle_radius_m=0.4215874761`、`safety_margin_m=0.10`、`min_clearance_m=0.5215874761`、`traversability_threshold=0.50`、`max_traversable_slope_deg=30.0`。
- 环境负责生成最终 `observed_safe_mask`；A* 不再二次 footprint inflation。
- 坐标统一为 `CellXY(x,y) <-> array[y,x] <-> cell-center world`；0.5m 栅格的 `(0,0)` 映射到 origin `+(0.25,0.25)`。adapter 只消费 A* 的 `path_cells`，不得透传当前存在半格偏差的 `path_world`。
- episode reset 在起点朝初始 theta 执行一次同合同 20m/90° LOS scan，不计 reward 和 step。

### 包结构与接口

新增根 `pyproject.toml` 和独立 `src/lunar_exploration_ppo/` 包，包含 `env/`、`policy/`、`ppo/`、`eval/`、`integrations/`、`configs/`、`workflows/`、`utils/`。新包不得依赖 `model_explorer`、legacy `scripts/xunce_*` 或修改 `sys.path`；只有 planner adapter 可导入 `path_planner`。

核心接口固定为：

```text
ScenarioSource.load(key) -> ScenarioBundle
GridGeometry.cell_to_world_center(cell) -> WorldXY
SensorUpdater.reveal(truth, observed_state, poses) -> ObservationDelta
ObservationBuilder.build(prior, observed_state, pose, action_set) -> PolicyObservation
FrontierGenerator.extract(observed_state, prior, pose) -> FrontierActionSet
PathPlannerAdapter.validate(safe_mask, start, target, theta) -> PlannerResult
LunarExplorationEnv.reset/step -> PolicyObservation | StepResult
CrossAttentionFrontierPolicy.forward(observation) -> PolicyForwardOutput
sample_action(output, mask, deterministic) -> ActionSample
RolloutBuffer.finalize(last_values) -> RolloutBatch
PPOTrainer.update(batch) -> PPOUpdateMetrics
CheckpointManager.save_complete/load_last_complete
Evaluator.evaluate(method, scenario_set) -> EvaluationSummary
```

`PolicyObservation` 固定为：prior 7 channels、coverage summary 8 channels、local crop 8 channels、frontier features 22 fields、pose 6 fields。`ObservationBuilder` 的类型和参数中不得出现 truth 或 `coverable_mask`。

### 网络与 PPO

- global/local CNN 均自适应池化到 `16x16`，各 256 tokens，context 总数 513。
- `D=128`、2 层 cross-attention、4 heads、FFN=512、action hidden=128、dropout=0。
- 使用 GroupNorm/LayerNorm，不使用 BatchNorm、RNN、candidate self-attention。
- `Q=frontier_tokens`，`K,V=global+local+pose context`。
- theta cos head 初始 bias=1，kappa 初始约 0.1；零范数与近零范数必须 fail-safe，不能只依赖 epsilon。
- 分布、logprob、GAE、ratio、KL 和 loss reduction 始终 FP32；v1 禁用 AMP。
- effective minibatch 固定 256；物理 microbatch 从 Smoke=8、Standard=4 开始，Stage 3 按 `[4,8,16,32]` 选择显存低于 10.1GiB 的最大安全值并冻结。
- 每个 env 收集 128 条 trainable transition，共 1024 条；诊断型空候选 reset 不占额度。
- `stagnation_done` 是真正 terminal；下一状态无候选时，上一条动作 transition 必须闭合 terminal，禁止伪造 logprob。
- penalty 保存为正幅值，但 reward 中严格减去；成功谓词统一为 `>=0.99`。
- baseline 的 theta 固定使用候选 `recommended_theta`。

## Artifact 与 Gate 合同

每阶段输出到 `D:/xunce/out/ppo_frontier/<run_id>/sN/`：

```text
config.json
summary.json
routing.json
manifest.json
report.md
metrics.jsonl
phase-state.jsonl
review.json
approval.json
gate.json
```

审批前不得生成 `approval.json` 和 `gate.json`。状态只能按以下顺序转换：

```text
machine_passed
-> awaiting_independent_review
-> awaiting_human_approval
-> approved
-> next stage
```

`gate.json` 绑定 Goal、Stage、Git tree、config/data/environment/checkpoint/review/manifest hashes和授权的下一 Stage。任何输入、代码或 artifact 漂移都会使 gate 失效；禁止 `--force` 跳阶段。

### Task 1: Foundation Gate — 环境、包装和合同冻结

**Foundation fixed values and boundaries:**

- `synthetic_source_kind=synthetic_terrain_obstacle_proxy/v1`
- `physical_obstacle_cells_written=false`
- `value_prior_source=constant_neutral/v1`
- `coordinate_convention=world_xy_grid_col_row_cell_center/v1`
- `theta_convention=radians_world_x_ccw_normalized_minus_pi_to_pi/v1`
- `CellXY(x,y) <-> array[y,x]`，0.5m 栅格 `(0,0)` 的 world center 为 origin `+(0.25,0.25)`。
- `vehicle_radius_m=0.4215874761`
- `safety_margin_m=0.10`
- `min_clearance_m=0.5215874761`
- `traversability_threshold=0.50`
- `max_traversable_slope_deg=30.0`
- `sensor_model_id=path-tangent-plus-endpoint-theta-fov-90-range-20m-los/v1`
- 初始 observation 为 reset 时从起点、初始 theta 执行一次同合同 20m/90° LOS scan，不计 reward 和 step。
- 默认 output root 为 `D:/xunce/out/ppo_frontier`；不得把 checkpoint、数据、训练输出或环境锁写入 Git。
- 新包不得依赖 `model_explorer`、任何 `scripts/xunce_*`，不得修改 `sys.path`；只有未来的 `integrations.path_planner_adapter` 可以导入 `path_planner`。
- Foundation 不实现 Stage 1 环境语义，不修改 legacy runner/default policy，不连接 executor，不启动 canary。

**Files:**

- Create: `pyproject.toml`
- Create: `src/lunar_exploration_ppo/__init__.py`
- Create: `src/lunar_exploration_ppo/configs/__init__.py`
- Create: `src/lunar_exploration_ppo/configs/schema.py`
- Create: `src/lunar_exploration_ppo/utils/__init__.py`
- Create: `src/lunar_exploration_ppo/utils/artifact_io.py`
- Create: `src/lunar_exploration_ppo/workflows/__init__.py`
- Create: `src/lunar_exploration_ppo/workflows/gates.py`
- Create: `src/lunar_exploration_ppo/integrations/__init__.py`
- Create: `src/lunar_exploration_ppo/env/__init__.py`
- Create: `src/lunar_exploration_ppo/policy/__init__.py`
- Create: `src/lunar_exploration_ppo/ppo/__init__.py`
- Create: `src/lunar_exploration_ppo/eval/__init__.py`
- Create: `scripts/run_ppo_foundation_preflight.py`
- Create: `configs/ppo_highres_frontier_foundation_v1.json`
- Create: `tests/ppo_highres_frontier/test_foundation.py`
- Modify: `scripts/bootstrap_env.py`
- Modify: `.github/workflows/platform-compatibility.yml`
- Modify: `.gitignore`

**Step 1: Write failing Foundation contract tests**

覆盖 wheel/editable install、无临时 `PYTHONPATH` import、固定 config schema、独立 import 边界、Windows 长路径、原子 JSON/checkpoint、真正 append+fsync JSONL、SHA-256 manifest、严格 gate 状态机、漂移失效、禁止 force，以及 `device=cuda` 不静默回落。

**Step 2: Run the focused test and confirm RED**

```powershell
python -m pytest tests/ppo_highres_frontier/test_foundation.py -q
```

**Step 3: Implement the minimum standalone foundation**

- 根包可 build/wheel/editable install；package discovery 只覆盖 `src/lunar_exploration_ppo`。
- 配置 schema 冻结 data/proxy、坐标/轴向、初始 observation、安全常量、输出根和 device 合同；未知字段 fail closed，派生 `min_clearance_m` 必须一致。
- 新 `ArtifactStore` 独立于 legacy runner：Windows-safe path、路径长度 warning `>180`、`>=240` fail、同目录临时文件+flush+fsync+atomic replace 写 JSON/checkpoint、JSONL 每条 append+flush+fsync、SHA-256 manifest。
- Gate 骨架只允许规定状态转换，绑定 Goal/Stage/Git tree/config/data/environment/checkpoint/review/manifest hashes；缺失或漂移 fail closed；不提供 `--force`。
- runner 只加载配置、执行 preflight、写轻量 artifacts；核心语义位于包内。
- bootstrap editable install 包含根包；CI 在 Windows/Ubuntu Python 3.12 安装根 wheel 并执行 CPU import/contract smoke。

**Step 4: Run focused and isolation tests**

```powershell
python -m pytest tests/ppo_highres_frontier/test_foundation.py -q
python -m pytest tests/test_bootstrap_env.py -q
```

**Step 5: Export and upgrade the authorized training environment**

- 在 `D:/xunce/env-backups/ppo_frontier/<timestamp>/` 导出 Conda explicit lock、environment YAML、pip freeze 和 torch 诊断。
- 将 `D:/conda_envs/lunar-explorer` 中 `torch 2.12.1+cpu` 替换为同版本 `torch 2.12.1+cu130`，不创建 C 盘模型或缓存。
- 验证 Python 3.12、`torch==2.12.1+cu130`、CUDA 可用、GPU forward/backward 有限；`device=cuda` 时不允许 CPU fallback。

**Step 6: Build/install/verify**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m build --wheel
D:/conda_envs/lunar-explorer/python.exe -m pip install --force-reinstall --no-deps <wheel>
D:/conda_envs/lunar-explorer/python.exe -c "import lunar_exploration_ppo"
```

分别在独立进程复核起点基线：root 8 项、A* 7 项、model-explorer 93 项及 13 个 subtests；不得合并为同一 pytest 进程。

**Step 7: Generate machine/review evidence and stop**

机器验收通过后输出 `config.json`、`summary.json`、`routing.json`、`manifest.json`、`report.md`、`metrics.jsonl`、`phase-state.jsonl`、`review.json`，状态停在 `awaiting_human_approval`；不生成 approval/gate，不提交。

**Approval commit:** `build: establish ppo frontier foundation`

### Task 2: Stage 1 — Smoke 环境闭环

实现 map state、0.5m proxy generator、传感器 ray casting、exact coverable mask、reachability、A* adapter、reward/done、progress 和环境闭环。覆盖坐标半格、GeoTIFF y 轴、unknown 禁行、单次安全膨胀、路径切线观测、endpoint theta、0.99 边界、stagnation/no-candidate terminal；完成设计中的 16 项验收与 10 个连续 Smoke episode。修改未观测 truth 不得改变 observation/candidate，成功观测后才允许变化。审查 truth 泄漏、planner 安全、reward 符号、terminal/bootstrap。

用户批准的 proxy fixture 增补要求 Smoke、Standard 与 Kilometer 的程序化 highres truth 含空间零散的岩石与陨石坑；陨石坑由坡度/可通行度判定，训练密度均衡混合 low/medium/high，Smoke 固定 medium，并沿用 hard-obstacle/slope-blocked 的 2D LOS。Stage 1 的详细 TDD、provenance、gate 和重审步骤见 `docs/superpowers/plans/2026-07-10-rock-crater-proxy-fixture.md`，设计合同见 `docs/superpowers/specs/2026-07-10-rock-crater-proxy-fixture-design-addendum.md`。

**Approval commit:** `feat: add smoke exploration environment`

### Task 3: Stage 2 — 完整 Observation 与 Frontier Generator

实现固定 channel/feature 顺序、local crop padding、summary aggregation、regular/irregular segment、recommended theta、observed-only potential gain、score-first top-M。建立按父级 ROI/macro-tile 先分割再裁剪的 Standard 场景目录：700 train、150 validation、150 test、64 空间隔离 unseen；同一父 ROI 不得跨 split。Rasterio 读取 affine/CRS；Standard 保留原生 4m prior，Kilometer 聚合 8m，记录 provenance。程序化 highres truth 使用 `procedural_lunar_rock_crater_proxy/v1`：train 对 low/medium/high 密度各 `1/3`，validation/test/unseen 按场景数近似等量分层且任意两档数量差不超过 1；proxy seed 在父 ROI split 后派生。验收 shape/order/finite、空集、overflow、稳定排序、NPZ round-trip、truth-mutation、密度分层与 split overlap=0。

**Approval commit:** `feat: add frontier observation pipeline`

### Task 4: Stage 3 — 网络 Forward 与动作分布

实现独立 `cross_attention_frontier_policy/v1`、masked categorical、Von Mises theta、确定性 eval、value pooling。性质测试覆盖 padding 不变性、候选排列等变性、无效候选零概率/零梯度、负索引拒绝、theta 周期与边界、零 raw norm 有限梯度、CPU/CUDA logprob 复算。CPU 误差 `<=1e-6`、CUDA `<=1e-5`；无 NaN/inf；deterministic action 完全一致。GPU 门：9.0GiB 预警、10.1GiB 硬停；batch-1 forward p95 Smoke `<=50ms`、Standard `<=100ms`。按 `[4,8,16,32]` 冻结最大安全 microbatch。

**Approval commit:** `feat: add cross-attention frontier policy`

### Task 5: Stage 4 — Rollout Buffer 与 PPO Update

实现 8 个 spawn-safe vector env、主进程批量 GPU inference、`[T=128,E=8]` snapshot buffer、GAE、联合 ratio、clipped policy/value loss、frontier entropy、KL early-stop、梯度裁剪和原子 checkpoint。old logprob/value、policy hash、candidate snapshot hash 缺失或不一致均 fail closed。保存模型、optimizer、RNG、normalizer、sampler、vector-env episode state、best record、配置 hash；只恢复最后完整 update。验收 1024 trainable transitions、3 次 Smoke update、初始 ratio≈1、post-clip grad norm `<=0.500001`、KL>0.03 early-stop、buffer 清空、save/load action 一致。Tiny-overfit 4 个一步 context、3 seeds、最多 200 updates，正确 frontier 概率 `>=0.95`、theta 误差 `<=0.1rad`、value MSE 降低 `>=90%`。

**Approval commit:** `feat: add on-policy ppo training`

### Task 6: Stage 5 — 公平 Baseline Evaluator

实现 random、nearest、max-gain、gain-over-cost 与 deterministic PPO。所有方法共用场景、候选集、recommended theta、planner、sensor、coverable denominator、预算和 seed。输出统一 metrics、coverage curve、固定 seed bootstrap 95% CI、fairness/leakage audit；完成设计中的 22 项验收，随机 baseline 固定 seed 可复现，确定性方法稳定 tie-break。

**Approval commit:** `feat: add fair baseline evaluation`

### Task 7: Stage 6 — Standard 正式训练与评估

顺序运行 5 个独立训练 seed，每个 100 PPO updates；每 10 次做 16 episode validation。每 seed 保存 latest、50 步 periodic、validation best；全局 best 按 success rate、mean final coverage、较早 seed/update 排序。冻结后执行 64 test、64 unseen 和四 baseline；test 不得选择或改变 checkpoint。可恢复状态机在 update 1/10/50/100 自动执行数学、显存、checkpoint、lineage 审计，异常立即停门。nonfinite/mask/snapshot/stale-policy 违规必须为 0；开始前 D 盘空闲 `>=100GiB`，运行中 `<50GiB` 停门；RSS 16GiB 预警、20GiB 硬停。报告分别给出系统与性能结论。

**Approval commit:** `feat: add standard ppo training workflow`

### Task 8: Stage 7 — Kilometer 稀疏压力测试

使用 Stage 6 全局 best checkpoint，eval-only、无 fine-tune、`max_steps=512`。先 8 episode 预检，再 32 test 与 32 unseen，至少与 gain-over-cost 同环境同预算比较；test/unseen 的 low/medium/high rock-crater 密度按场景数近似等量分层。2048² highres 只留在环境；policy 保持 128² prior、192² local crop、2048 sparse frontiers、513 context tokens。exact coverable mask 按数据/起点/安全/sensor/proxy-catalog hash 预计算和校验缓存。硬门：VRAM `<=10.1GiB`、RSS `<=20GiB`、policy forward p95 `<=250ms`、candidate+prefilter p95 `<=8s`、无泄漏/OOM/NaN。结论仅表述为 proxy-based kilometer stress test。

**Approval commit:** `feat: add kilometer stress evaluation`

### Task 9: Stage 8 — 最终证据包

汇总配置、checkpoint manifest、训练/覆盖曲线、baseline 表、失败分类、leakage audit、复现命令和全部 gate。`final_report.md` 区分真实低分辨率 prior、程序化 highres proxy、当前已观测 highres state。运行完整分项目测试、wheel 安装测试和 Stage 1–8 dry-run/reproduction smoke。使用 `superpowers:requesting-code-review` 审查 merge-base 到当前树，修复并重审全部 Critical/Important。只提交源码、配置、测试、轻量文档与索引。

**Approval commit:** `docs: package ppo frontier results`

## Foundation 起点基线

开始 Foundation 前已验证：root 8 项通过、A* 7 项通过、model-explorer 93 项及 13 个 subtests 通过。各子项目必须分进程测试，避免 import-isolation 测试互相污染。
