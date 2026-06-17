# visual-workbench

Artifact-first evidence workbench for the parent `lunar-path-planning` repository.

This is a local fourth-subproject candidate. It is intentionally not added to
`.gitmodules` yet because no remote repository is configured. When a remote is
available, it can be converted into a sibling Git submodule following the
existing parent repository pattern.

## Scope

- Reads allowlisted artifact roots only:
  - parent `outputs/`
  - `dev-platform-constraints/outputs/`
  - `model-explorer/outputs/`
  - `path-planner/outputs/`
  - paths supplied through `VISUAL_WORKBENCH_ARTIFACT_ROOTS`
- Detects and displays:
  - `model-explorer-contract/v1`
  - `path-planner-sidecar/v1`
  - `path-planner-route/v1`
  - `path-feedback-manifest/v1`
  - `path-feedback-summary/v1`
  - `model-explorer-experiment/v1`
- Allows only white-listed `dry-run` and `validate` command execution.
- Rejects complete experiment `run` requests.

## Mission-Control Direction

The current UI direction is mission-first, not tool-first. The primary
navigation stages are, in order: 环境测绘, 目标捕获, 路线制导, 可达确认,
风险复核, 任务简报.

The workbench uses a dark mission map/status surface with a light progressive
evidence panel. Raw evidence is hidden by default and revealed through
progressive evidence disclosure so operators can follow the mission state first
and drill into source artifacts only when needed.

Stage-local research entry points are Evidence Trace, Map Replay, and Validate.
Validate remains limited to `dry-run 或 validate`; `完整 run 不会` be triggered
from the UI.

Scope guards remain unchanged: no PPO/training/full run, no network/action
space/default A* changes, no Ackermann-feasible trajectory claim, and no
IRIS/GCS/path-planner diagnostics as release or training proof.

## Non-goals

- No PPO, training, staged release, or full experiment run from the UI.
- No changes to network, action space, default A*, or existing subproject logic.
- No Ackermann-feasible trajectory claim.
- No promotion of IRIS/GCS/path-planner diagnostics into training or release proof.

## Run

Backend:

```powershell
cd visual-workbench
python -m visual_workbench.api
```

Optional demo artifacts when local `outputs/` is empty:

```powershell
$env:VISUAL_WORKBENCH_ARTIFACT_ROOTS='C:\Users\77634\.codex\worktrees\ca49\lunar-path-planning\visual-workbench\fixtures\demo-artifacts'
python -m visual_workbench.api
```

Frontend:

```powershell
cd visual-workbench\web
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`.

## Verify

```powershell
cd visual-workbench
python -m pytest
cd web
npm test
npm run build
```

To keep download caches off C by default:

```powershell
$env:PIP_CACHE_DIR='D:\CodexDownloads\pip-cache'
$env:npm_config_cache='D:\CodexDownloads\npm-cache'
```
