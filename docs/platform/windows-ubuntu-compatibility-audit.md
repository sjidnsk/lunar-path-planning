# Windows 与 Ubuntu 兼容性审计

自动 CI 覆盖 Windows/Ubuntu non-Drake（Python 3.12），可选 Ubuntu Drake profile 只在手动触发且 `pydrake` 可用时运行。

保留的父仓库检查包括 Stage6 标准配置、bootstrap、platform smoke/matrix、G2 输入合同和 mainline surface guards；子模块检查只覆盖 `path-planner` 与 `dev-platform-constraints`。平台矩阵不调用历史 `run_stage.py`，不运行 release/canary/path-feedback 或 Stage15--18。

`tests/ppo_highres_frontier/test_foundation.py` 保留冻结 worktree 身份与原子证据写入合同，属于离线身份绑定审计，不属于 Windows/Ubuntu 跨平台 CI。跨平台基础检查只验证父包导入、CPU device 合同以及上述可移植主线测试，不降低 Foundation gate 的 fail-closed 安全边界。

`tests/ppo_highres_frontier/test_stage6_standard_config.py` 同时包含可移植的 Stage6 配置合同，以及读取固定 `D:/xunce/out/.../s5/gate.json` 的 Stage5 不可变证据回放。后者显式标记为 `external_evidence`，Windows/Ubuntu 平台 profile 只过滤该标记；不带标记过滤器的本地完整测试仍会执行证据哈希链、篡改拒绝和 canonical JSON 检查，证据缺失时继续 fail-closed。

`scripts/bootstrap_env.py` 是跨平台 bootstrap 入口；`scripts/run_platform_smoke.py` 与 `scripts/run_platform_validation_matrix.py` 分别提供轻量 smoke 和矩阵入口。默认规划仍为 Python grid A*；Drake 不是 Windows 支持条件。
