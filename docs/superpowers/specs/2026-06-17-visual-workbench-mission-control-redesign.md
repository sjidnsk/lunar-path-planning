# Visual Workbench Mission-Control Redesign

## Summary

The next `visual-workbench` phase should move from a basic artifact browser to a
mission-first lunar survey control room. The selected direction is **A+B**:
build a visually stronger showcase MVP while preserving the key research
workbench entry points needed for artifact traceability and lightweight
validation.

The product should answer this first question on entry:

> 当前月面巡视任务走到哪里了？

It should not open as a generic data dashboard. The first screen should look and
behave like a lunar mission status surface, with evidence available through
progressive disclosure rather than exposed as raw tables by default.

## Approved Direction

- Product stance: mission-first evidence workbench.
- Visual stance: split dark/light interface.
- Primary surface: dark mission map and task status area.
- Secondary surface: light evidence explanation panel.
- Navigation model: task-first navigation, not tool-first navigation.
- Evidence model: progressive panel, showing conclusion first and raw evidence
  only after expansion.
- Build strategy: showcase MVP plus critical research entry points.

## Mission Stage Rail

The main stage rail uses mission language directly:

1. 环境测绘
2. 目标捕获
3. 路线制导
4. 可达确认
5. 风险复核
6. 任务简报

Each stage maps to existing project evidence without renaming the underlying
schemas:

| Mission stage | Primary evidence | User-facing meaning |
| --- | --- | --- |
| 环境测绘 | `model-explorer-contract/v1`, `path-planner-sidecar/v1` | The terrain, cost, passability, and candidate context are known. |
| 目标捕获 | `model-explorer-contract/v1`, experiment outputs | Candidate goals have been ranked or selected. |
| 路线制导 | `path-planner-route/v1` | A route candidate exists and can be replayed on the map. |
| 可达确认 | `path-feedback-manifest/v1`, `path-feedback-summary/v1` | The route has passed or failed lightweight feedback validation. |
| 风险复核 | summaries, rejection reports, reason codes | Known blockers, caveats, and failure reasons are visible. |
| 任务简报 | Markdown reports, summaries, presentation view | The current evidence can be communicated without changing meaning. |

## Home Layout

The home screen should be rebuilt around four regions:

1. Left rail: mission-first navigation using the six mission stages.
2. Top rail: phase status strip showing completed, current, blocked, and pending
   stages.
3. Main dark canvas: lunar map, route overlay, selected goal, route status, and
   current mission location.
4. Right light panel: progressive evidence explanation for the selected stage.

The current `Overview / Evidence Browser / Map & Route / Path Feedback /
Experiment Matrix / Presentation` structure should become secondary. Those
capabilities remain, but they are accessed from the relevant mission stage
instead of competing as top-level pages.

## Progressive Evidence Panel

The selected stage opens a right-side evidence panel with three levels:

### Level 1: Mission Judgment

Visible by default.

- Current stage name.
- Plain-language status.
- One-sentence meaning.
- Highest-risk blocker.
- Next recommended action.

Example:

```text
路线制导
候选路径已生成，等待可达确认。
当前路径可在地图上回放，但还不能作为训练或发布证明。
```

### Level 2: Why It Is Credible

Shown below the judgment or after a small expansion.

- Required schemas found.
- Artifact status summary.
- Latest modified time.
- Known reason codes or blocker count.
- Source paths in compact form.

### Level 3: Raw Evidence

Explicitly opened by the user.

- Artifact summary.
- Raw JSON/Markdown.
- Schema version.
- Full path.
- Copy/open affordance where safe.

Raw evidence should not dominate the first screen.

## Critical Research Entrypoints

Each mission stage should expose three compact tools:

1. Evidence Trace
   - Show related artifacts.
   - Open summary/raw.
   - Compare status and schema coverage.

2. Map Replay
   - Show map layers relevant to the stage.
   - Overlay goals, raw path, smoothed path, and optimized path.
   - Keep legend visible and do not rely on color alone.

3. Validate
   - Trigger only white-listed `dry-run` and `validate` commands.
   - Show command result, stderr, parsed JSON, and recovery hints.
   - Reject full `run` before subprocess execution.

These entries keep the MVP useful for development without turning the first
screen back into a dense tool menu.

## Visual System

Use a split dark/light visual system:

- Dark mission area:
  - background near `#020617`;
  - surfaces near `#0f172a`;
  - map grid and route overlay with high contrast;
  - blue for current stage, green for passed, amber for pending/risk, red for
    blocker.

- Light evidence area:
  - white or slate-50 surfaces;
  - dark text for long reading;
  - subtle borders;
  - compact status rows.

Typography should feel technical and precise. A future implementation can use
Fira Sans for body text and Fira Code or a similar mono face for technical
labels, but the first implementation may stay with local system fonts if web
font loading is not desired.

## Interaction Model

- Clicking a mission stage updates the map focus and the evidence panel.
- Clicking a map goal or route segment opens the related evidence panel section.
- Presentation mode hides raw paths, long file paths, stderr, and low-level
  diagnostics unless explicitly expanded.
- Research mode keeps all evidence affordances available.
- The same artifact data powers both modes; presentation mode must not rewrite
  metric meaning.

## Data and API Needs

The existing API can support the first redesign, but the frontend should add a
derived mission model:

```text
artifacts[] -> schema detector -> stage grouping -> stage status -> panel model
```

The derived model should be frontend-side at first unless backend performance
or duplication becomes a problem.

Potential future backend additions:

- `GET /api/mission/stages`
- `GET /api/mission/stages/{stage_id}`
- `GET /api/artifacts/{artifact_id}/related`

These are not required for the next MVP if the frontend can derive stage state
from `/api/artifacts`.

## Accessibility and Responsiveness

The redesign must preserve the previous verification bar:

- No horizontal overflow at 375, 768, 1024, and 1440 widths.
- Keyboard focus must be visible.
- Stage state cannot be conveyed by color alone.
- Map legend must include labels and non-color distinctions where practical.
- Progressive panel sections need accessible buttons with expanded/collapsed
  state.
- Motion should be subtle and respect reduced-motion preferences.

## Testing and Verification

Minimum verification for the redesign:

```powershell
cd visual-workbench
python -m pytest
cd web
npm test
npm run build
```

Frontend tests should cover:

- stage rail renders the six mission stages;
- selecting a stage updates the evidence panel;
- raw evidence is hidden by default and expands on action;
- presentation mode hides low-level diagnostics;
- command entry points still reject full `run`;
- responsive smoke checks for no horizontal overflow.

Backend tests should remain focused on artifact allowlisting, schema detection,
raw reads, and command white-list enforcement.

## Non-Goals

This redesign must not:

- start PPO, training, staged release, or complete experiment `run`;
- modify `network/action space/default A*`;
- replace responsibilities of `dev-platform-constraints`, `model-explorer`, or
  `path-planner`;
- claim Ackermann-feasible trajectory;
- treat IRIS/GCS/path-planner diagnostics as training approval, release proof,
  or performance proof;
- remove artifact-first traceability in favor of a purely decorative dashboard.

## First Implementation Slice

The next implementation plan should target this narrow slice:

1. Add a mission stage model in the frontend.
2. Replace the current top-level navigation with mission-first navigation.
3. Rebuild the home screen around dark map canvas plus light evidence panel.
4. Add progressive evidence panel behavior.
5. Keep Evidence Trace, Map Replay, and Validate as stage-local entry points.
6. Preserve existing backend API and tests unless a small endpoint addition
   removes clear duplication.

This slice is intentionally smaller than full productization. It should create
the control-room experience first, while keeping research operations reachable.
