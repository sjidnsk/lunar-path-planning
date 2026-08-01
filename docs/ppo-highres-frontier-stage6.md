# PPO High-resolution Frontier Stage 6

Stage 6 执行 Standard v1 正式训练与最终评估。唯一配置入口是
`configs/ppo_highres_frontier_stage6_v1.json`，运行前必须验证冻结的 Stage 5 Gate、
Stage 4 checkpoint 与 policy-state 哈希；任何权威文件或源码漂移都应失败关闭。

## R1 failed-run rollover

`s6-standard-single-r1-20260718T062833Z` 的 R1 是不可变的失败证据：它只完成
四个完整 update，Update 5 在形成完整 checkpoint/update commit 前失败。Update 4 的
checkpoint、optimizer、policy、RNG 或 vector state 均不会被携带；不得重写其 lineage、
恢复其运行或把它描述为成功训练。经新的完整审查授权后，新的正式 run 从 Update 1 重新开始，
使用冻结的 Stage 4 initialization；不会预先硬编码该未来 run id。

## Standard v1 合同

- 场景为 128 m × 128 m，局部栅格为 256 × 256（0.5 m/格），先验图为
  32 × 32（4 m/格），局部裁剪为 96 × 96，候选槽位为 1024。
- 每回合最多 128 步，停滞窗口为 16，成功条件是精确 coverable 集覆盖率不低于
  0.99；synthetic terrain 只能作为 proxy，不能作为物理障碍真值。
- 安全分解固定为 `vehicle_radius_m=0.4215874761`、`safety_margin_m=0.10`、
  `min_clearance_m=0.5215874761`、`traversability_threshold=0.50`、
  `max_traversable_slope_deg=30.0`；配置、环境、公平评估和 Stage 6 checkpoint
  payload/manifest 必须逐项一致，不能只校验总 clearance。
- 训练使用 8 个环境、每环境 128 条 transition、batch 1024、有效 minibatch
  256、microbatch 32、每次 4 个 PPO epoch；FP32，AMP 关闭。
- Stage 6A 只训练 seed `20260716`，完整执行 100 次 update；每 10 次 update
  做 16 回合 validation，并冻结 validation-best checkpoint。它是
  `single_seed_system_closure/v1`，只证明系统闭环，不构成跨 seed 性能结论；额外
  3--5 个 seed 只有用户明确要求才追加，且不阻塞 Stage 6 Gate、Stage 7 或 Stage 8。
- Stage 7 直接消费 Stage 6A seed `20260716` 的 frozen validation-best checkpoint，
  不等待可选的多-seed扩展。
- 最终 test 与 unseen 各 64 个场景；PPO 和四个冻结 baseline 在相同场景与随机种子上
  各跑 64 回合，总计 640 回合。baseline 的角度使用候选推荐角度，PPO 使用策略角度。

## 资源与执行边界

Stage 6 正式训练当前只支持 Windows。正式入口在初始 review authorization 验证后、
读取配置或创建任何输出之前，强制固定全部执行输入：canonical config、完整
`STAGE6_SOURCE_PATHS`（包括 `test_foundation.py`）、DEM 与 slope provenance、冻结的
Stage 5 gate/approval/review/manifest/Stage 4 checkpoint/checkpoint manifest，以及
Stage 6 review authorization 的 5 个 evidence members。每个文件保持一个
`GENERIC_READ`、仅 `FILE_SHARE_READ`、`OPEN_REPARSE_POINT`、不可继承且 single-link 的
持久句柄；每个不同父目录的完整祖先目录 guard 也保持到正式调用最终返回。这样 config
load、run lease、machine preflight、spawn worker、Rasterio、Stage 4 checkpoint load、
100-update 训练和 terminal/preterminal recovery 均处于同一锁生命周期内。进入锁后先重算
完整 execution identity 并逐项匹配已审查记录；生命周期中按 identity/size/link/parent
重验，最终返回前再做一次完整哈希。任何 writer、hardlink、reparse、父链/路径/字节漂移
都失败关闭，pin 对象本身不写入 lineage，只有 authorization handle 的 canonical record
进入 lineage。

历史 `test_stage1_smoke_env_r1.py` 已从仓库、Stage 1 reviewed source set 和
`STAGE6_TEST_SOURCE_PATHS` 移除。任何仍绑定该历史测试的 Stage 1 或 Stage 6 authorization
或 source identity 都必须因 source-set 漂移失败关闭；正式执行或训练只能在新的独立 review
完成后重新获得授权，不能更新或伪造旧记录的哈希来继续执行。

workflow 只在上述锁、review authorization、Stage 5 authority 与完整 input pin 均有效时，
签发一个进程内、不可复制、不可序列化且绑定 exact run/root/config/evidence 的 opaque
execution capability。正式训练入口、production backend 以及四个 terminal recovery 写接口
在入口、关键写入边界和退出时都重新验证同一个 capability；scope 退出、lease 丢失或任一
绑定漂移后立即撤销并失败关闭。直接调用内部入口、伪造 mapping，或只复用旧 capability
都不得创建目录、artifact、checkpoint 或 worker。
每个受保护 operation 都持有进程内引用；撤销先进入 `revoking`、拒绝新 operation，并等待
全部在途 operation 结束后才从 registry 移除，因此 scope 退出不能与持久写入发生竞态。
execution identity、resource segment、checkpoint、trace、metrics、retention 与 canonical
publication 在各自写边界再次校验同一 capability。

`os.name != 'nt'` 时，强 pin 的公开 acquisition API 在调用时明确失败关闭，不能以 POSIX
advisory lock 冒充正式固定。Ubuntu CPU CI 仍可导入该模块并执行 import smoke，但不得运行
Stage 6 正式训练。Windows 的锁保证面向普通同机进程和项目 runner，不声称抵御管理员、
内核级攻击者或能绕过 NT 文件共享语义的主体。

正式启动前 D 盘可用空间必须至少 100 GiB；运行中低于 50 GiB 停止。RSS 达到
16 GiB 告警、达到或超过 20 GiB 停止；CUDA VRAM 超过 9 GiB 告警，preflight
达到或超过 10.1 GiB 停止，runtime 同样在达到或超过 10.1 GiB 时停止。预检必须覆盖真实
Standard reset/step、CUDA forward/backward、Windows spawn
collector 集成，并记录 Standard 场景构建、coverable 计算及缓存命中相关分段计时。

RSS 使用 `process_tree_lifecycle_peak_current_sum/v1`：从正式入口进入后，持续覆盖训练、
最终评估、待提交验收 payload 的语义重放以及终态资源采样；周期采样主进程及全部后代进程
在同一时刻的 current RSS 总和，并保留生命周期峰值。完成上述 workload 后，后端先在
monitor 运行时执行一次显式终态采样，再停止并等待监控线程退出，随后原子追加终态资源行；
只有该终态行通过重放后，才允许发布 summary、routing、三份报告、success phase 和唯一最终
manifest。因此持久化证据覆盖停止前的显式样本及其间任何并发样本，且 `machine_passed`
不会早于终态资源验收。这些步骤属于证据封装边界，禁止再启动模型 workload 或子进程。资源审计同时记录
root PID、采样次数、最新进程数和峰值时进程数；机器重放要求 root PID 稳定，采样次数与
树峰值不得回退。瞬时父进程 RSS 不能作为 Stage 6 运行证据。

## 数学、公平性与恢复证据

- 每次 PPO update 的数学审计绑定全部 1024 条 transition、candidate snapshot 列表、
  policy-state hash，以及每次 forward、loss 和梯度步骤的 FP32 有限性、联合 logprob
  分解误差、ratio、KL、loss 和裁剪后梯度范数；缺失或哈希不一致均失败关闭。
- 最终评估的每个 split/method 使用独立 attempt 目录。trace 与 summary 只有在
  `stage6_final_eval_commit/v1` 同时绑定二者哈希后才可发布为 canonical；中断后只恢复
  已密封 attempt，半发布文件不会被当作完成结果。trace 中每条
  `(scenario_id, scenario_seed, terrain_seed, start_pose_seed, evaluation_seed)` 必须与
  fairness 场景调度逐行一致，即使重新计算 trace/summary/commit 哈希也不能绕过。
- `stage6_standard_parallel_fairness_audit/v2` 逐动作记录 `PolicyObservation`、候选 mask、
  selected index、方法选择规则和角度来源。test 与 unseen 分别执行五方法 cohort 重放，
  要求场景/seed 调度、环境合同和 observation schema 完全一致；baseline 的 theta 必须
  从所选候选的 recommended-theta sin/cos 重建。
- leakage audit 扫描冻结 observation schema 和运行时 selector 输入字段；任何 truth、
  coverable mask 或隐藏 high-resolution 字段进入决策输入都会阻断机器验收。
- `resource_audit.jsonl` 的 v2 终态行绑定追加前 JSONL 的 SHA-256 与大小；唯一最终 manifest
  直接绑定包含该终态行的完整资源日志。最终 verifier 还会从原始 metrics、checkpoint receipts
  和 immutable lineage 精确重建 summary、routing 与三份报告，拒绝未知字段、报告漂移以及
  通过重算 manifest 重新合法化语义篡改。
- JSONL 写入使用 pending 事务、fsync、descriptor/path identity 与 no-follow 能力；run
  lease 另使用 Linux abstract AF_UNIX guard 或 Windows named mutex，防止 lease 文件被
  rename/recreate 后形成双写者。该边界面向同机项目 runner 与可测试路径替换，不声称
  抵御管理员、root、恶意内核或隔离 IPC namespace 的攻击者。

独立审查通过并获得 Stage 6 正式执行授权前，只运行预检和轻量测试，不启动完整
1 × 100 正式训练。外部授权的 `formal_run_id` 必须与本次正式 run-id 完全一致；正式
入口会在创建 run root、lease、preflight 或其他正式 artifact 之前验证授权。恢复时仍须
传入并重新验证同一外部 authorization，禁止复制或生成替代授权，也不存在 force、skip、
fake、provider 或 consume 绕过。同一授权不能换用另一个即使格式合法的 run-id。

正式运行及恢复命令：

```powershell
$runId = '<与授权文件 formal_run_id 完全一致的正式 run-id>'
$reviewAuthorization = 'D:/xunce/review/<review-package>/launch-authorization.json'
D:/conda_envs/lunar-explorer/python.exe scripts/run_ppo_stage6_standard.py `
  --run-id $runId `
  --review-authorization $reviewAuthorization
```

所有运行产物写入 D 盘。性能优势仅在 PPO 的 95% CI 下界严格高于
gain-over-cost baseline 的 95% CI 上界时成立；未建立性能优势不阻断系统机器验收。
