# Retired Route Cleanup

目标：在 `codex/retired-route-cleanup` 上收缩父仓库，只保留当前主线与必要共享基础设施。执行依据为 `D:/xunce/out/route_retire_preflight/candidate-manifest.json`（SHA-256 `da9f23c3af83c3506143338a41e1c1a8f8f09dc4f59d4a4bac5adea160f6371c`）及用户 2026-08-02 的统一清理授权。

## Global Constraints

- 保留 Stage 6 高分辨率前沿 PPO、默认 grid A*、中期 G1/G2/G3、`path-planner` v3、`dev-platform-constraints` 与仍被主线导入的共享 helper。
- 退役 Stage18--25、Stage26、path-feedback / 早期策略实验、`model-explorer` 与 `visual-workbench`。
- 不运行训练、完整实验、checkpoint 发布、executor 或 canary；只运行测试和只读审计。
- 不使用递归或通配符删除命令；每次删除必须指向一个精确 tracked 路径。不得触碰现有未跟踪用户文件。
- `model-explorer`、`visual-workbench` 仅移除父仓库 gitlink 与 `.gitmodules` stanza，不递归删除本地目录。
- 每个任务独立提交并接受审查；保留路线基线失败即修复后再继续。

### Task 1: Decouple Retired Submodules And Parent Path-Feedback Entrypoints

- 移除 `model-explorer`、`visual-workbench` gitlink 与 `.gitmodules` stanza，同时保留本地目录。
- 从 bootstrap、platform smoke/matrix 及其测试中移除这两个子模块和 parent path-feedback 的必需绑定。
- 保持 `path-planner`、`dev-platform-constraints`、默认 A*、Stage 6、G1/G2/G3 行为不变。
- 更新覆盖上述入口的测试并运行 parent platform、Stage 6 与 G1/G2/G3 聚焦基线。

### Task 2: Remove Manifest-Listed Path-Feedback And Early-Policy Files

- 从固定 candidate manifest 中选取 `matched_rules` 包含 `path_feedback` 的全部 `retire_candidate` 精确路径。
- 删除对应 runner/config/test/doc，并移除 `configs/stage_registry.json` 中只服务于这些路径的记录。
- 删除或收敛仍只引用这些路径的早期策略入口；保留任何主线引用。
- 运行 parent platform 与保留路线聚焦基线。

### Task 3: Remove Manifest-Listed Stage26 Files

- 删除固定 candidate manifest 中全部 Stage26 `retire_candidate` 精确路径，不因历史引用而保留。
- 清理 Stage26 registry 与入口引用，但保留共享 artifact IO/path helpers。
- 运行 Stage 6、G1/G2/G3、artifact helper 与 planner adapter 基线。

### Task 4: Remove Manifest-Listed Stage18--25 Files

- 删除固定 candidate manifest 中全部 Stage18--25 `retire_candidate` 精确路径。
- 清理 registry 与只服务于退役链的引用；保留当前 G1/G2/G3 与平台/规划合同。
- 运行 Stage 6、G1/G2/G3、parent platform 基线。

### Task 5: Collapse Remaining Historical Surface To Mainline Entries

- 对剩余 tracked 文件执行引用审计；仅删除可确认属于上述退役路线的未分类 dependent 文件。
- 更新 `README.md`、`AGENTS.md`、`docs/project-route-map.md`、`docs/route-retirement-manifest.md` 与阶段索引，使其描述清理后的真实树。
- 保留最小退役记录、备份定位和恢复说明，不保留已失效的运行入口。

### Task 6: Final Mainline Verification

- 运行 Stage 6/default A*、G1/G2/G3、parent platform、`path-planner` Python/C++ 与 `dev-platform-constraints` 基线。
- 运行 `tests/test_route_retirement_preflight.py`、`git diff --check`、删除范围审计和 broken-reference 检查。
- 输出 D 盘 ignored runtime artifact 的精确人工清理清单；不使用递归删除命令。
- 进行整分支审查，确认没有删除主线、共享 helper 或用户未跟踪文件。
