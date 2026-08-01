# Task 2: Path-feedback 与早期策略入口清理报告

## 范围与输入

- 固定 manifest：`D:/xunce/out/route_retire_preflight/candidate-manifest.json`
- 已验证 SHA-256：`da9f23c3af83c3506143338a41e1c1a8f8f09dc4f59d4a4bac5adea160f6371c`
- 基线：`9e400307896999b62e75e0ca1e006805b86285c0`
- 精确主集合：`classification=retire_candidate` 且 `matched_rules` 含 `path_feedback` 的 29 个 tracked 路径。

## 删除结果

主集合按单一明确路径逐项通过 `apply_patch` 删除：

| 分类 | 数量 |
| --- | ---: |
| `configs/path_feedback_*` | 17 |
| `scripts/*path_feedback*` | 8 |
| `tests/*path_feedback*` | 4 |
| 主集合合计 | 29 |

同时删除了 7 个仅把上述 batch runner/config 串接到旧 canary/sequential 链的 closure shell 脚本，以及 4 个仅测试该已删除 shell runner 的 canary 测试；合计物理删除 40 个 tracked 文件。没有删除 Stage18--26 manifest candidates、Stage6、G1/G2/G3、`path-planner`、`dev-platform-constraints`、shared artifact helpers、`model-explorer`、`visual-workbench` 或未跟踪文件。

## Registry、平台和测试收敛

- 从 `configs/stage_registry.json` 移除了 `path-feedback-validation` 与 `path-feedback-batch-validation`。
- 收敛 `docs/platform/windows-ubuntu-compatibility-audit.md` 的已删除 runner/registry 入口。
- 移除已删除 Windows 兼容测试在历史 platform plan 中的命令行入口，并删除 Bash allowlist 中已不存在的唯一例外。
- 新增 `tests/test_retired_path_feedback_cleanup.py`：持续检查 29 个精确路径均不存在，且 registry/platform matrix/CI 不重新暴露该入口。

## RED / GREEN

RED：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_retired_path_feedback_cleanup.py -q
2 failed
- manifest-listed absence audit 列出全部 29 个仍存在的路径
- registry audit 发现 path-feedback stage key
```

GREEN（项目 Python `D:/conda_envs/lunar-explorer/python.exe`，并把当前 worktree 的 `path-planner/src` 放在 `PYTHONPATH` 首位）：

```text
pytest tests/test_retired_path_feedback_cleanup.py tests/test_route_retirement_preflight.py \
  tests/test_platform_stage_runner.py tests/test_bootstrap_env.py \
  tests/test_bootstrap_ubuntu_conda.py tests/test_platform_smoke.py \
  tests/test_no_new_python_bash_dependencies.py tests/test_platform_validation_matrix.py -q
52 passed, 1 skipped

pytest tests/ppo_highres_frontier/test_stage6_standard_config.py \
  tests/ppo_highres_frontier/test_g1_safe_start_fallback.py \
  tests/test_xunce_mid_dual_contracts.py tests/test_xunce_mid_dual_g1_coverage.py \
  tests/test_xunce_mid_dual_g2_inputs.py tests/test_xunce_mid_dual_g2_planning_time.py \
  tests/test_xunce_mid_dual_g3_closed_loop.py -q
297 passed, 2 skipped

python scripts/run_platform_validation_matrix.py --profile windows-non-drake --dry-run
passed; emitted only retained platform, Stage15--18, path-planner and dev-platform-constraints commands
```

所有 pytest 临时目录使用 `D:/xunce/basetemp/retired-route-task2-*`。

## 残余引用与裁决

- 22 个 2026-06 的历史 plans/specs 仍以文本方式记录已删除文件名；它们不是 registry/CI/platform 运行入口。按主任务的收敛边界，它们与更广的早期策略历史一起留给 Task5，避免本任务扩展为历史文档大范围删除。
- `tests/test_route_retirement_preflight.py` 保留 `scripts/run_path_feedback_validation.py` 作为 retirement-policy 分类样例；它不调用该脚本，且 preflight 契约已通过。
- `tests/test_policy_decision_robustness_analysis.py` 与 `tests/test_sample_quality_training_application.py` 在 fixture 元数据中保留 `configs/path_feedback_batch_dataset_v1.json` 字符串；本任务删除了其中真正执行已删除 runner 的 compatibility case。剩余 fixture 语义属于早期策略测试，留给 Task5。
- 更广的早期策略 runners（例如 `scripts/run_policy_gated_sequential_canary_rollout.py`）仍可能引用已退役 runner；它们不在固定 manifest 的 29 项直接集合内，且主任务要求不无限扩张，因此标记给 Task5，不能作为保留路线入口使用。

## 自审与 concerns

- post-change audit：29 个 manifest 路径均不存在；registry 内无 `path-feedback` key；`git diff --check` 通过。
- 项目解释器初始从一个陈旧 worktree 导入了 `path_planner`，导致 G2 planning-time test 缺少 `formal_request_codec`；显式将当前 `path-planner/src` 置于 `PYTHONPATH` 首位后，G2 focused test 44 passed，完整保留路线合同组 297 passed / 2 skipped。
- commit：当前 `HEAD` 的 Task 2 cleanup commit。
