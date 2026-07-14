# Stage 3 Cross-Attention Policy

Stage 3 只验证 `cross_attention_frontier_policy/v1` 的网络 forward、masked categorical、
候选条件 Von Mises theta、value pooling 与 CUDA 资源门禁。它不执行 PPO 更新，不保存或发布
checkpoint，也不代表任务性能优于任何 baseline。

机器执行必须显式消费由外部控制器持久化的 Stage 2 verified gate，并将运行产物写入 D 盘：

```powershell
D:\conda_envs\lunar-explorer\python.exe scripts/run_ppo_highres_frontier_stage3.py `
  --run-id <run_id> `
  --stage2-gate <existing_stage2_gate_path>
```

成功执行只到达 `machine_passed -> awaiting_independent_review`。runner 不创建 review、approval
或 gate，不连接 executor，不启动 canary，也不替换默认策略。
