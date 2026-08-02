# Stage 6 精确 Coverable Mask 持久缓存设计补充

## 目标与证据

本补充实现用户批准的方案 B：不改变 `compute_coverage_masks()` 的任何算法语义，将其逐位相同的结果预计算到 D 盘，并让 Stage 6 Standard 训练只读加载已校验缓存。

正式 preflight 已记录：单场景 `coverable_env_init=129.548168600013s`，8 个 spawn worker 首次 `spawn_reset=211.77540649997536s`，而场景构建和普通 reset 分别仅约 `0.154s` 与 `0.109s`。Update 49 有 46 条 terminal transition，因此新场景的 exact coverable-mask 重算是当前明确的主要热点。

## 不变合同

- 官方算法仍为 `exact_reachable_safe_pose_range_los/v1`，三个数组必须与当前 `compute_coverage_masks()` 逐位相同。
- 不改变 20m sensor range、`min_clearance_m=0.5215874761`、30° slope、0.50 traversability、LOS DDA、start、proxy、reward、done、PPO、候选或 checkpoint schema。
- coverable mask 仍只用于环境侧覆盖率分母和审计；不得进入 `PolicyObservation`、frontier features 或 action selection。
- 缓存缺失、损坏、key/manifest/source/hash 不匹配时正式训练 fail closed，禁止静默重算。
- 预热工具可以计算缺失项；正式 runner 只能读缓存，不写缓存。
- Update 50 attempt1 继续废弃；完成源码、缓存、review 和 ordinal6 授权后，从 Update 49 完整 checkpoint 启动 Update 50 attempt2、resource segment 7。
- 不引入新依赖，不实现异步 collector，不修改 sensor 方案 A 已通过双审的四个冻结文件。

## 缓存身份

每个 entry 的 canonical key schema 为 `exact_coverable_mask_cache_key/v1`，至少绑定：

```text
scenario_hash
start_cell_xy
geometry(width,height,resolution_m,origin_x_m,origin_y_m)
algorithm_id=exact_reachable_safe_pose_range_los/v1
los_model=two_dimensional_grid_line_of_sight/v1
sensor_range_m
min_clearance_m
max_slope_deg
traversability_threshold
cache_format=exact_coverable_mask_npz/v1
```

key 使用 `ArtifactStore.canonical_json_bytes()` 后取 SHA-256。entry 路径固定为：

```text
<cache_root>/entries/<sha256[0:2]>/<sha256>.npz
```

NPZ 使用未压缩、`allow_pickle=False` 的数组格式，包含：

```text
safe_free_mask        bool [H,W]
reachable_safe_mask   bool [H,W]
coverable_mask        bool [H,W]
metadata_utf8         uint8 [N]
```

metadata 绑定 canonical key、每个数组的 dtype/shape/SHA-256、coverable cell count、原算法 metadata 和 entry schema。entry 通过同目录临时文件与 exclusive atomic publish 写入；已存在的 entry 必须加载验证，禁止覆盖。

## 全目录 Manifest 与只读加载

预热完成后生成 `coverage-cache-manifest.json`，schema 为 `stage6_exact_coverable_cache_manifest/v1`，绑定 catalog SHA、split counts、1064 个场景的 scenario id/key SHA/entry path/entry SHA/size、聚合 entry-set SHA、生成环境和完整性统计。

Stage 6 production source identity 固定绑定 manifest 的 SHA-256 和字节数。每个 worker 启动时只读加载 manifest；每次绑定新场景时：

1. 从 scenario 与安全合同重算 canonical key。
2. 要求 manifest 中该 scenario 只有一个精确匹配 entry。
3. 使用安全路径读取固定 entry bytes，验证文件 SHA/size。
4. `np.load(..., allow_pickle=False)`，逐项验证 schema/key/shape/dtype/array hash/count/metadata。
5. 构造只读 `CoverageMasks`，再由环境复制为 episode-local masks。

任何异常均抛出稳定的 cache contract error，不回退到计算。

## 预热工作流

- 固定预热 1064 个 Standard 场景：700 train、150 validation、150 test、64 unseen。
- 使用 Windows `spawn`，最多 16 个计算 worker；parent 是唯一 progress/manifest writer。
- worker 只调用现有 `StandardScenarioFactory.build()` 与 `compute_coverage_masks()`，然后返回 entry bytes 和摘要；不修改正式 run artifacts。
- parent 按场景 ID 稳定顺序接收/记录结果，entry 原子 exclusive publish；恢复时验证已有 entry 后跳过。
- 输出目录为 `D:/xunce/review/<prewarm_run_id>/`，至少包含 `config.json`、`progress.jsonl`、`summary.json`、`manifest.json`、`report.md`；缓存根位于 `D:/xunce/cache/ppo_frontier/<formal_run_id>/coverage-v1/`。
- D 盘低于 100GiB 不启动，低于 50GiB 停门；parent+worker RSS 超过 20GiB 停门；nonfinite、worker failure、重复/缺失 scenario、entry mismatch 立即失败。

## 验收

- 单元测试覆盖 key 稳定性、NPZ round-trip、逐位数组等价、损坏/截断/错 key/错 manifest/路径逃逸/重复 publish fail closed。
- Standard env 同一场景由 legacy compute 与 cache load 构造后，coverage metadata、三个 mask、reset observation、candidate、首条 deterministic step 完全一致。
- leakage 测试确认 cache/coverable mask 不进入 ObservationBuilder 参数、PolicyObservation 或 frontier 生成。
- prewarm dry-run 小集合可恢复；正式 1064-entry manifest 无缺失、无重复、split 数精确。
- cache-hit `coverable_env_init` 同机代表场景 median 必须不高于 5s，且相对 `129.548168600013s` 至少 20×；目标不高于 1s。
- 方案 A 的 21 项 focused regression 与 2.8103205847754213× benchmark 保持有效。
- ordinal6 必须同时绑定 sensor 提速证据、cache source set、1064-entry manifest、性能/等价证据、fresh 双审和 Update 49/50 cutover。

## 声明边界

该缓存只改变确定性派生数据的计算时机与存储位置，不改变算法或训练样本语义。cache-hit 提速不等于 PPO 性能优势；单 seed 仍不能支持多 seed 结论。
