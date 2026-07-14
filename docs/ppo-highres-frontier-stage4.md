# Stage 4: Rollout 与 PPO update

Stage 4 在 `Smoke v1` 上验证训练系统闭环：8 个 Windows `spawn` 环境各采集
128 条 trainable transition，主进程使用 CUDA FP32 policy inference，随后执行固定
PPO update、完整 checkpoint save/load 和 deterministic action bit-exact 复验。机器门连续
执行三次 update，并运行 4-context × 3-seed tiny-overfit。update 1 后机器门会销毁原
trainer/vector runtime，再通过 `resume_training_from_last_checkpoint` 把 model、AdamW、
Python/NumPy/Torch RNG、trainer update step、normalizer、scenario sampler 与 8 个 worker
episode/sampler state 应用到 fresh runtime；update 2 从恢复后的 current observation 继续，
不会先 reset 丢弃 worker state。

checkpoint 发布先在 `.pending-*` 目录内独占写入并 fsync `checkpoint.pt`、
`manifest.json` 与 `complete.json`，然后一次目录 rename 原子发布。rename 前故障只暴露
上一完整 update；rename 后故障时新 update 已完整可加载，下一 update 可继续保存；任何
已完成 update 均不可覆盖。

运行入口：

```powershell
D:/conda_envs/lunar-explorer/python.exe scripts/run_ppo_highres_frontier_stage4.py `
  --config configs/ppo_highres_frontier_stage4_v1.json `
  --run-id <unique-run-id> `
  --stage3-gate D:/xunce/out/ppo_frontier/s3-task4-cifix-final-20260713T213739Z/s3/gate.json
```

输出固定写入 `D:/xunce/out/ppo_frontier/<run-id>/s4/`，包含 canonical config、
summary、routing、manifest、JSONL、审计证据和三个 machine-evidence checkpoint。
`evidence/training_resume_audit.json` 绑定恢复 receipt、全部 runtime-state hash、恢复后的
collection policy hash、N+1 update 前后 policy hash 与 optimizer hash。
状态只到 `machine_passed -> awaiting_independent_review`；runner 不签发 Stage 4
review、approval 或 gate。

本阶段只证明 rollout、PPO、checkpoint 和恢复机制闭环，不证明任务性能优势。机器
checkpoint 仅为审计证据，不发布、不替换 default policy，也不连接 executor 或启动
canary。
