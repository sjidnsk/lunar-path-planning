# PPO Stage 6 Single-Seed Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The Stage gate contract overrides skill defaults: no task-level commit; the only Stage 6 commit is created after machine acceptance, fresh dual review, and explicit human approval.

**Goal:** 修复 Standard unsafe-start 根因，并把 Stage 6 正式执行收敛为一个完整的 100-update单-seed系统闭环，同时把额外 seeds 移出 Stage 1–8 关键路径。

**Architecture:** 场景工厂在 proxy 生成前从 0.5m upsampled DEM 计算 exact-safe 起点候选，保持 deterministic seed 与 6m start protection。Stage 6 配置和状态机只接受 seed `20260716`，但保留全部 validation、checkpoint、test/unseen、baseline、恢复和审计语义；可选多-seed扩展不进入本 runner。

**Tech Stack:** Python 3.12、NumPy 2.2.6、Rasterio 1.5.0、PyTorch 2.12.1+cu130、Pydantic、pytest、Windows spawn。

## Global Constraints

- 只使用 `D:/conda_envs/lunar-explorer/python.exe`；所有 pytest temp、cache 和大型输出写 D 盘。
- 不恢复或修改 R1/R2；不启动重复 runner；正式修复使用新 run-id。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。
- synthetic terrain 始终为 `synthetic_terrain_obstacle_proxy/v1`，`physical_obstacle_cells_written=false`。
- 不改变 PPO、network、reward、planner、observation schema、1024 transition/update、FP32 或 AMP=false 合同。
- 额外 3–5 seeds 只有用户明确要求才执行，不阻塞 Stage 7/8。
- 所有实现遵循 TDD；fresh implementer 完成后由 fresh规格 reviewer 和 fresh质量 reviewer 独立审查；Critical/Important 必须修复并重审。

## Failed R1 rollover boundary

`s6-standard-single-r1-20260718T062833Z` 的 R1 是不可变的失败证据：它只保留四个
完整 update，Update 5 在完整 checkpoint/update commit 前失败。Update 4 的
checkpoint、optimizer、policy、RNG 或 vector state 均不会被携带，不得恢复或重写其
lineage。后续经新的完整审查授权，新的正式 run 从 Update 1 重新开始，并从冻结的 Stage 4
initialization 启动；未来 run id 不能预先硬编码。额外 seeds 仍只有用户明确要求才追加，
且不阻塞 Stage 6 Gate、Stage 7 或 Stage 8。

---

### Task 1: 冻结单 Seed 与可选扩展合同

**Files:**
- Modify: `docs/superpowers/specs/2026-07-09-ppo-highres-frontier-map-exploration-design.md:3015`
- Modify: `docs/superpowers/plans/2026-07-10-ppo-highres-frontier-map-exploration.md:226`
- Modify: `docs/ppo-highres-frontier-stage6.md`
- Modify: `configs/ppo_highres_frontier_stage6_v1.json`
- Modify: `src/lunar_exploration_ppo/configs/stage6.py`
- Test: `tests/ppo_highres_frontier/test_stage6_standard_config.py`

**Interfaces:**
- Consumes: `Stage6Config.training` 与既有 Stage 5 authority。
- Produces: `training.seeds == (20260716,)`，并在文档中冻结 `single_seed_system_closure/v1` 与 user-only optional extension。

- [ ] **Step 1: 写 RED 配置测试**

将 frozen assertion 改为：

```python
assert config.training.seeds == (20260716,)
assert config.training.updates_per_seed == 100
```

并断言配置中不存在自动追加 seed、潜力阈值或 Stage 6B runner 字段。

- [ ] **Step 2: 运行 RED**

```powershell
$env:TEMP='D:/xunce/pytest/s6a-config-red/tmp'
$env:TMP=$env:TEMP
D:/conda_envs/lunar-explorer/python.exe -B -m pytest -p no:cacheprovider `
  --basetemp D:/xunce/pytest/s6a-config-red/base `
  tests/ppo_highres_frontier/test_stage6_standard_config.py -q
```

Expected: frozen training seed assertion fails because current config contains five seeds.

- [ ] **Step 3: 最小修改配置和文档**

只把 Stage 6A训练 seed 改为 `20260716`；100 updates、10-update validation cadence、16 validation episodes、64 test、64 unseen 和四 baseline不变。设计主文和主计划明确 Stage 7 使用 Stage 6A best，Stage 6B 仅用户触发且不阻塞后续 Stage。

- [ ] **Step 4: 运行 GREEN 与 UTF-8 验证**

运行 Step 2 命令；再用 Python `Path(...).read_text(encoding='utf-8')` 读取三份中文文档并断言包含“只有用户明确要求才追加”。

### Task 2: TDD 修复 deterministic exact-safe 起点

**Files:**
- Modify: `src/lunar_exploration_ppo/env/scenario_catalog.py:223-308,460-471`
- Test: `tests/ppo_highres_frontier/test_stage2_catalog.py`
- Test: `tests/ppo_highres_frontier/test_stage6_standard_env.py`

**Interfaces:**
- Consumes: 32×32 real prior、256×256 upsampled base DEM、`ScenarioCatalogRecord.start_pose_seed_hex`。
- Produces: `_standard_start_pose(record, prior, base_height, geometry) -> PoseXYTheta`，其 cell 在相同 30°/0.50/0.5215874761 合同下 exact-safe。

- [ ] **Step 1: 写精确 RED 回归**

选取 `train/scenario-0506`，先断言其 `prior_traversability >= 0.75` 候选为空，再 `StandardScenarioFactory.build(record)`，从最终 proxy truth 重建一次 final safe mask并断言：

```python
start = bundle.start_pose.cell
assert final_safe[start.y, start.x]
```

同时遍历 1064 条 catalog记录的 base DEM，断言每条至少有一个 exact-safe 内圈低分辨率中心。当前代码应在 `scenario-0506` 断言失败。

- [ ] **Step 2: 运行 RED**

```powershell
D:/conda_envs/lunar-explorer/python.exe -B -m pytest -p no:cacheprovider `
  --basetemp D:/xunce/pytest/s6a-start-red/base `
  tests/ppo_highres_frontier/test_stage2_catalog.py `
  tests/ppo_highres_frontier/test_stage6_standard_env.py -q `
  -k "exact_safe_start or scenario_0506"
```

Expected: `train/scenario-0506` start is false in final safe mask.

- [ ] **Step 3: 实现 source-level选择**

在 proxy 生成前对 `base_height` 调用 `derive_physical_slope_deg(..., spacing_m=0.5)`，构造 finite/slope/traversability free mask，再调用 `apply_clearance_once`。将 4m 内圈中心映射为 `CellXY(low_x * 8 + 4, low_y * 8 + 4)`，先筛 `prior>=0.75 and exact_safe`，为空时筛 `exact_safe`；仍为空则抛出稳定 `ValueError`。禁止 env reset跳过记录。

- [ ] **Step 4: 运行 GREEN 与目录审计**

运行 Step 2 命令，并执行只读审计：1064/1064 有候选；26 个 fallback 场景最终 proxy 26/26 safe；`scenario-0506` deterministic repeated build 的 start/scenario hash一致。

### Task 3: TDD 改造单 Seed 状态机与全局 best

**Files:**
- Modify: `src/lunar_exploration_ppo/ppo/standard_training.py:33,80-83,120-174,1735-1880,2208-2312`
- Modify: `src/lunar_exploration_ppo/workflows/stage6.py`
- Test: `tests/ppo_highres_frontier/test_stage6_training.py`
- Test: `tests/ppo_highres_frontier/test_stage6_execution.py`
- Test: `tests/ppo_highres_frontier/test_stage6_workflow.py`

**Interfaces:**
- Consumes: frozen `Stage6Config.training.seeds == (20260716,)`。
- Produces: 100 ordered transactions、10 validation records、单 seed validation best作为 frozen global best、完整 final evaluation schedule。

- [ ] **Step 1: 写 RED 调度测试**

断言：

```python
assert FROZEN_SEEDS == (20260716,)
assert len(build_standard_training_transactions(config)) == 100
assert sum(tx.validation_episodes == 16 for tx in transactions) == 10
assert transactions[-1].commit_states[-1] == "seed_20260716_complete"
assert select_global_best([seed_best], configured_seeds=(20260716,)) == seed_best
```

并断言任何额外 seed 都 fail closed；final evaluation仍为 PPO test/unseen 加四 baseline×两个 split。

- [ ] **Step 2: 运行 RED**

```powershell
D:/conda_envs/lunar-explorer/python.exe -B -m pytest -p no:cacheprovider `
  --basetemp D:/xunce/pytest/s6a-schedule-red/base `
  tests/ppo_highres_frontier/test_stage6_training.py `
  tests/ppo_highres_frontier/test_stage6_execution.py `
  tests/ppo_highres_frontier/test_stage6_workflow.py -q `
  -k "single_seed or training_schedule or global_best"
```

- [ ] **Step 3: 最小修改 frozen schedule**

把 `FROZEN_SEEDS` 与所有明确要求 five completed seeds 的检查改为单 seed合同；保持 seed-local fresh initialization、sampler lane seeds、transaction hash chain、checkpoint restore suffix 和 tie-break代码通用。不得添加 Stage 6B自动入口。

- [ ] **Step 4: 运行 GREEN**

运行 Step 2 命令，另执行全部 `test_stage6_training.py` 和相关 execution/workflow测试。

### Task 4: 恢复、lineage 与报告语义

**Files:**
- Modify: `src/lunar_exploration_ppo/ppo/standard_training.py`
- Modify: `src/lunar_exploration_ppo/workflows/stage6.py`
- Modify: `scripts/run_ppo_stage6_standard.py`
- Test: `tests/ppo_highres_frontier/test_stage6_execution.py`
- Test: `tests/ppo_highres_frontier/test_stage6_workflow.py`

**Interfaces:**
- Consumes: 新单-seed config SHA、prospective tree、R2失败证据，以及绑定 exact
  `formal_run_id` 的外部 review authorization。
- Produces: 新 run-id 的严格 lineage；summary/report含单-seed限制；Stage 6 route直接指向 Stage 7。

- [ ] **Step 1: 写 RED lineage/report测试**

断言旧 R2 config/source hash不能被恢复接受；public workflow 与 CLI 必须提供
`--review-authorization`，且在创建 run root、lease、preflight 前校验授权中的 exact
`formal_run_id`。每次恢复重新验证同一外部 authorization；同一授权换用另一个合法
run-id 必须零副作用拒绝。新 summary/report包含 `single_seed_system_closure/v1`、
`cross_seed_performance_conclusion=false`、`optional_seed_extension_blocks_next_stage=false`，
routing 的下一 Stage是 Stage 7。

- [ ] **Step 2: 运行 RED 后实现最小字段**

只增加机器可审计的单-seed结论字段和稳定中文/英文报告文本；不创建 Stage 6B runner，不读取 R2 checkpoint，不覆盖 R2文件。

- [ ] **Step 3: 运行恢复与 artifact回归**

运行 Stage 6 execution/workflow/durable JSONL/checkpoint/resource测试；模拟半写 checkpoint 和 update 1前崩溃，确认只恢复最后完整 update且旧 lineage fail closed。

### Task 5: Fresh 双审与正式 Stage 6A

**Files:**
- Evidence only: `.superpowers/sdd/` 与运行时生成的 `D:/xunce/out/ppo_frontier/$runId/s6/`

**Interfaces:**
- Consumes: 通过回归的冻结 source/config package。
- Produces: machine-passed Stage 6A artifacts、fresh规格/质量双审、`awaiting_human_approval` 证据。

- [ ] **Step 1: 主 agent 分进程回归**

分别运行 Stage 2 catalog/env、Stage 4 collector/checkpoint/trainer、全部 Stage 6测试；使用唯一 D 盘 basetemp并确认无残留 worker。运行 CUDA batch forward/backward和正式 preflight，确认 GPU、RSS、D盘门。

- [ ] **Step 2: 冻结 review package并 fresh双审**

先冻结唯一正式 run-id。review package必须绑定该 exact `formal_run_id`、base commit、
完整 changed path set、prospective tree、config/data/environment hashes。两个 reviewer
均需给出 Critical/Important/Minor计数；Critical/Important非零则回到 fresh implementer
修复并重审。

- [ ] **Step 3: 启动唯一正式 run**

确认系统中没有 Stage 6 runner后，以授权绑定的同一 run-id运行：

```powershell
$runId = '<与授权文件 formal_run_id 完全一致的正式 run-id>'
$reviewAuthorization = 'D:/xunce/review/<review-package>/launch-authorization.json'
D:/conda_envs/lunar-explorer/python.exe scripts/run_ppo_stage6_standard.py `
  --run-id $runId `
  --review-authorization $reviewAuthorization
```

禁止恢复 R2。若同一正式 run 中断，必须用同一 `$runId` 和同一外部 authorization
重跑上述命令，入口会重新消费并校验授权；不得提供 force/skip/fake/provider/consume
参数。持续监控 seed/update/transaction、stdout/stderr、进程树、GPU、RSS、D盘和
checkpoint receipts。

- [ ] **Step 4: 正式 artifact审计和最终 fresh双审**

机器通过后核对 config、summary、routing、manifest、report、metrics、phase/job/resource/checkpoint artifacts及 hash chain；确认单 seed描述性结论、无跨 seed声明、route授权 Stage 7。

- [ ] **Step 5: 停在人工门**

写入 `review.json` 并转换到 `awaiting_human_approval`；不得写 `approval.json`/`gate.json`，不得提交。向用户提交 Stage 6A证据，只有明确批准 Stage 6 Gate后才创建唯一提交 `feat: add standard ppo training workflow` 并授权 Stage 7。
