# Visual Workbench H1 Mission Cockpit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `visual-workbench` from the current mission-control MVP into the approved H1 mission cockpit: stage narrative, mission status KPIs, interactive map replay, right-side judgment/tools, and bottom evidence-chain drawer.

**Architecture:** Keep the existing FastAPI artifact API unchanged. Add a frontend-only cockpit domain model that derives KPIs, map selection state, replay frames, and evidence-chain data from existing artifacts, sidecar, route, and selected mission stage. Compose the UI from focused React components so `App.tsx` remains orchestration only.

**Tech Stack:** React, TypeScript, Vite, Vitest, Testing Library, lucide-react, Canvas 2D, FastAPI, Pydantic, pytest, Playwright/Edge smoke checks.

---

## Scope Guard

This plan implements:

`docs/superpowers/specs/2026-06-17-visual-workbench-h1-mission-cockpit-design.md`

It assumes the current worktree already contains the mission-control MVP:

- `MissionStageRail`
- `MissionMap`
- `MissionEvidencePanel`
- `StageTools`
- shared `types.ts`
- frontend mission stage derivation

This plan is an incremental H1 upgrade. It must not:

- start PPO, training, staged release, or full experiment `run`;
- modify `network/action space/default A*`;
- replace responsibilities of `dev-platform-constraints`, `model-explorer`, or `path-planner`;
- claim Ackermann-feasible trajectory;
- treat IRIS/GCS/path-planner diagnostics as training approval, release proof, or performance proof;
- add backend mission endpoints unless the implementation explicitly discovers that frontend derivation is not viable.

## File Structure

Create or modify these frontend files:

- `visual-workbench/web/src/domain/missionCockpit.ts`
  - Derive H1 KPIs, replay frames, selected map object, layer state defaults, and evidence-chain model.

- `visual-workbench/web/src/domain/missionCockpit.test.ts`
  - Unit tests for KPI derivation, replay frames, and evidence-chain missing-schema/guard behavior.

- `visual-workbench/web/src/components/MissionStatusHeader.tsx`
  - Render H1 top status: project label, question, selected-stage summary, API status, mode status, and mode toggle.

- `visual-workbench/web/src/components/MissionKpiStrip.tsx`
  - Render evidence completeness, artifact count, replay frame/layer summary, and residual risk.

- `visual-workbench/web/src/components/MissionMapReplay.tsx`
  - Replace `MissionMap` in `App.tsx` for the main cockpit map.
  - Keep Canvas 2D drawing behavior, but add layer controls, timeline controls, selected map object callback, and stable accessible names.

- `visual-workbench/web/src/components/EvidenceChainDrawer.tsx`
  - Render selected map object, supporting artifacts, schema flow, missing schema, next safe action, and forbidden actions.

- `visual-workbench/web/src/components/StageToolPanel.tsx`
  - Move the current inline `StageToolPanel` out of `App.tsx`.
  - Give Evidence Trace, Map Replay, and Validate stronger H1-specific content.

- `visual-workbench/web/src/App.tsx`
  - Load data as today.
  - Manage selected stage, active tool, selected replay frame, selected map object, presentation mode.
  - Compose H1 cockpit layout.

- `visual-workbench/web/src/App.test.tsx`
  - Add integration tests for H1 cockpit layout, KPI strip, layer/timeline interaction, drawer update, tool toggles, and validate guard.

- `visual-workbench/web/src/styles.css`
  - Add H1 cockpit layout, map replay, KPI strip, drawer, responsive rules, focus states, and non-overflow safeguards.

Update documentation:

- `visual-workbench/README.md`
- `tests/test_visual_workbench_boundary.py`
- optionally append a short note to `docs/superpowers/specs/2026-06-17-visual-workbench-h1-mission-cockpit-design.md` only if implementation changes the design.

---

### Task 1: Add H1 Cockpit Domain Model

**Files:**
- Create: `visual-workbench/web/src/domain/missionCockpit.ts`
- Create: `visual-workbench/web/src/domain/missionCockpit.test.ts`
- Modify: `visual-workbench/web/src/types.ts`

- [ ] **Step 1: Add failing domain tests**

Create `visual-workbench/web/src/domain/missionCockpit.test.ts`:

```ts
import { describe, expect, test } from "vitest";

import type { Artifact, RoutePayload, SidecarPayload } from "../types";
import { deriveMissionStages } from "./missionStages";
import {
  DEFAULT_MAP_LAYERS,
  buildEvidenceChain,
  buildMissionCockpitKpis,
  buildReplayFrames,
  firstReplaySelection,
} from "./missionCockpit";

function artifact(id: string, schema_version: string, status = "passed"): Artifact {
  return {
    artifact_id: id,
    name: `${id}.json`,
    relative_path: `outputs/${id}.json`,
    kind: "json",
    schema_version,
    status,
    size_bytes: 100,
    modified_at: "2026-06-17T00:00:00Z",
  };
}

const artifacts = [
  artifact("contract", "model-explorer-contract/v1"),
  artifact("sidecar", "path-planner-sidecar/v1"),
  artifact("route", "path-planner-route/v1", "reachable"),
];

const sidecar: SidecarPayload = {
  schema_version: "path-planner-sidecar/v1",
  top_goals: [{ cell: [2, 2], reachable: true, utility: 0.91 }],
};

const route: RoutePayload = {
  schema_version: "path-planner-route/v1",
  reachable: true,
  geometric_path: [
    [0, 0],
    [1, 1],
    [2, 2],
  ],
  postprocess: {
    smoothed_path: [
      [0, 0],
      [1, 0.75],
      [2, 2],
    ],
  },
  trajectory_optimization_report: {
    optimized_path: [
      [0, 0],
      [1.2, 0.8],
      [2, 2],
    ],
  },
};

describe("mission cockpit model", () => {
  test("derives cockpit KPIs from selected stage and artifacts", () => {
    const stage = deriveMissionStages(artifacts).find((item) => item.id === "route-guidance");
    if (!stage) throw new Error("route-guidance stage missing");

    const kpis = buildMissionCockpitKpis(stage, artifacts, route);

    expect(kpis.evidenceCompletenessLabel).toBe("100%");
    expect(kpis.keyArtifactCount).toBe(1);
    expect(kpis.replayFrameLabel).toBe("t2");
    expect(kpis.riskLabel).toBe("中低");
  });

  test("builds replay frames from route points and goals", () => {
    const frames = buildReplayFrames(sidecar, route);

    expect(frames.map((frame) => frame.id)).toEqual(["t0", "t1", "t2"]);
    expect(frames[2]).toMatchObject({
      label: "目标接近",
      objectType: "path-segment",
      pathIndex: 2,
    });
    expect(firstReplaySelection(sidecar, route)).toMatchObject({
      objectType: "path-segment",
      frameId: "t2",
    });
  });

  test("builds evidence chain with missing feedback and forbidden actions", () => {
    const stage = deriveMissionStages(artifacts).find((item) => item.id === "route-guidance");
    if (!stage) throw new Error("route-guidance stage missing");

    const chain = buildEvidenceChain(stage, artifacts, firstReplaySelection(sidecar, route));

    expect(chain.supportingArtifacts.map((item) => item.schema_version)).toContain("path-planner-route/v1");
    expect(chain.missingSchemas).toContain("path-feedback-summary/v1");
    expect(chain.forbiddenActions).toEqual(["full run", "PPO", "training"]);
    expect(DEFAULT_MAP_LAYERS.optimizedPath).toBe(true);
  });
});
```

- [ ] **Step 2: Run the domain test to verify failure**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/domain/missionCockpit.test.ts
```

Expected: FAIL because `./missionCockpit` does not exist.

- [ ] **Step 3: Extend shared types**

Append these types to `visual-workbench/web/src/types.ts`:

```ts
export type MapLayerState = {
  rawPath: boolean;
  smoothedPath: boolean;
  optimizedPath: boolean;
  blocked: boolean;
};

export type SelectedMapObject = {
  objectType: "mission-stage" | "goal" | "path-segment";
  frameId: string;
  label: string;
  pathIndex?: number;
  cell?: [number, number];
  artifactId?: string;
};

export type ReplayFrame = SelectedMapObject & {
  timeLabel: string;
};

export type MissionCockpitKpis = {
  evidenceCompletenessLabel: string;
  keyArtifactCount: number;
  replayFrameLabel: string;
  riskLabel: string;
};

export type EvidenceChain = {
  selected: SelectedMapObject;
  supportingArtifacts: Artifact[];
  schemaFlow: Array<{ schema: string; status: "present" | "missing" }>;
  missingSchemas: string[];
  nextSafeAction: string;
  forbiddenActions: string[];
};
```

- [ ] **Step 4: Implement the cockpit domain model**

Create `visual-workbench/web/src/domain/missionCockpit.ts`:

```ts
import type {
  Artifact,
  EvidenceChain,
  MapLayerState,
  MissionCockpitKpis,
  ReplayFrame,
  RoutePayload,
  SelectedMapObject,
  SidecarPayload,
} from "../types";
import type { DerivedMissionStage } from "./missionStages";

export const DEFAULT_MAP_LAYERS: MapLayerState = {
  rawPath: true,
  smoothedPath: true,
  optimizedPath: true,
  blocked: true,
};

const FEEDBACK_SCHEMAS = ["path-feedback-manifest/v1", "path-feedback-summary/v1"];

export function buildMissionCockpitKpis(
  stage: DerivedMissionStage,
  allArtifacts: Artifact[],
  route: RoutePayload | null,
): MissionCockpitKpis {
  const required = stage.schemas.length || 1;
  const present = stage.schemas.filter((schema) =>
    allArtifacts.some((artifact) => artifact.schema_version === schema),
  ).length;
  const completeness = Math.round((present / required) * 100);
  const hasBlocked = stage.artifacts.some((artifact) =>
    ["blocked", "failed", "error", "rejected"].includes(artifact.status.toLowerCase()),
  );

  return {
    evidenceCompletenessLabel: `${completeness}%`,
    keyArtifactCount: stage.artifacts.length,
    replayFrameLabel: route ? `t${Math.max(0, normalizePath(route.geometric_path).length - 1)}` : "t0",
    riskLabel: hasBlocked || route?.reachable === false ? "高" : stage.artifacts.length ? "中低" : "证据不足",
  };
}

export function buildReplayFrames(sidecar: SidecarPayload | null, route: RoutePayload | null): ReplayFrame[] {
  const path = normalizePath(route?.geometric_path);
  if (path.length) {
    return path.map(([x, y], index) => ({
      objectType: "path-segment",
      frameId: `t${index}`,
      timeLabel: `t${index}`,
      label: index === 0 ? "起点" : index === path.length - 1 ? "目标接近" : "路径段",
      pathIndex: index,
      cell: [x, y],
      artifactId: "path-planner-route/v1",
    }));
  }

  const goal = sidecar?.top_goals?.find((item) => Array.isArray(item.cell));
  if (goal?.cell) {
    return [
      {
        objectType: "goal",
        frameId: "t0",
        timeLabel: "t0",
        label: "目标点",
        cell: goal.cell,
        artifactId: "path-planner-sidecar/v1",
      },
    ];
  }

  return [
    {
      objectType: "mission-stage",
      frameId: "t0",
      timeLabel: "t0",
      label: "任务阶段",
    },
  ];
}

export function firstReplaySelection(sidecar: SidecarPayload | null, route: RoutePayload | null): SelectedMapObject {
  const frames = buildReplayFrames(sidecar, route);
  return frames[frames.length - 1];
}

export function buildEvidenceChain(
  stage: DerivedMissionStage,
  allArtifacts: Artifact[],
  selected: SelectedMapObject,
): EvidenceChain {
  const schemas = Array.from(new Set([...stage.schemas, ...schemasForSelection(selected), ...FEEDBACK_SCHEMAS]));
  const schemaFlow = schemas.map((schema) => ({
    schema,
    status: allArtifacts.some((artifact) => artifact.schema_version === schema) ? "present" as const : "missing" as const,
  }));
  const supportingArtifacts = allArtifacts.filter((artifact) =>
    schemaFlow.some((item) => item.status === "present" && item.schema === artifact.schema_version),
  );
  const missingSchemas = schemaFlow.filter((item) => item.status === "missing").map((item) => item.schema);

  return {
    selected,
    supportingArtifacts,
    schemaFlow,
    missingSchemas,
    nextSafeAction: missingSchemas.includes("path-feedback-summary/v1")
      ? "进入可达确认，只执行 dry-run 或 validate。"
      : "汇总证据并进入风险复核。",
    forbiddenActions: ["full run", "PPO", "training"],
  };
}

function schemasForSelection(selected: SelectedMapObject): string[] {
  if (selected.objectType === "path-segment") return ["path-planner-route/v1"];
  if (selected.objectType === "goal") return ["path-planner-sidecar/v1"];
  return [];
}

function normalizePath(value: unknown): Array<[number, number]> {
  if (!Array.isArray(value)) return [];
  return value.flatMap((point) => {
    if (!Array.isArray(point) || point.length < 2) return [];
    const [x, y] = point;
    return typeof x === "number" && Number.isFinite(x) && typeof y === "number" && Number.isFinite(y)
      ? [[x, y] as [number, number]]
      : [];
  });
}
```

- [ ] **Step 5: Run the cockpit domain tests**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/domain/missionCockpit.test.ts
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

Run:

```powershell
git add visual-workbench/web/src/types.ts visual-workbench/web/src/domain/missionCockpit.ts visual-workbench/web/src/domain/missionCockpit.test.ts
git commit -m "feat: derive h1 cockpit view model"
```

---

### Task 2: Add Mission Status Header And KPI Strip

**Files:**
- Create: `visual-workbench/web/src/components/MissionStatusHeader.tsx`
- Create: `visual-workbench/web/src/components/MissionKpiStrip.tsx`
- Modify: `visual-workbench/web/src/App.tsx`
- Modify: `visual-workbench/web/src/App.test.tsx`

- [ ] **Step 1: Add failing integration test**

Append this test inside `describe("App", ...)` in `visual-workbench/web/src/App.test.tsx`:

```tsx
test("renders H1 mission status header and cockpit KPIs", async () => {
  render(<App />);

  expect(await screen.findByText("H1 任务驾驶舱")).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "当前月面巡视任务走到哪里了？" })).toBeInTheDocument();
  expect(screen.getByText("证据完整度")).toBeInTheDocument();
  expect(screen.getByText("关键 artifact")).toBeInTheDocument();
  expect(screen.getByText("回放帧")).toBeInTheDocument();
  expect(screen.getByText("剩余风险")).toBeInTheDocument();
  expect(screen.getByLabelText("API 状态：ok")).toHaveClass("connection-status");
});
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: FAIL because `H1 任务驾驶舱` and KPI strip are not rendered.

- [ ] **Step 3: Create `MissionStatusHeader`**

Create `visual-workbench/web/src/components/MissionStatusHeader.tsx`:

```tsx
import { PanelLeft, Presentation, Signal } from "lucide-react";

import type { DerivedMissionStage } from "../domain/missionStages";
import { getStageDescription, getStageLabel } from "./MissionStageRail";

type MissionStatusHeaderProps = {
  stage: DerivedMissionStage;
  health: string;
  presentationMode: boolean;
  onTogglePresentationMode: () => void;
};

export function MissionStatusHeader({
  stage,
  health,
  presentationMode,
  onTogglePresentationMode,
}: MissionStatusHeaderProps) {
  return (
    <header className="mission-status-header">
      <span className={health === "ok" ? "connection-status ok" : "connection-status"} aria-label={`API 状态：${health}`}>
        <Signal size={14} aria-hidden="true" />
        API {health}
      </span>
      <div className="mission-status-copy">
        <p className="section-label">lunar-path-planning / visual-workbench</p>
        <p className="cockpit-label">H1 任务驾驶舱</p>
        <h1>当前月面巡视任务走到哪里了？</h1>
        <p className="mission-summary">
          {getStageLabel(stage.id)}：{getStageDescription(stage.id)}
        </p>
      </div>
      <div className="mission-topbar-actions">
        <span className={presentationMode ? "mode-pill presentation" : "mode-pill research"}>
          <Presentation size={16} aria-hidden="true" />
          {presentationMode ? "演示模式" : "研发模式"}
        </span>
        <button
          className="button secondary"
          type="button"
          aria-label={presentationMode ? "切换到研发模式" : "切换到演示模式"}
          onClick={onTogglePresentationMode}
        >
          <PanelLeft size={16} aria-hidden="true" />
          {presentationMode ? "研发视图" : "演示视图"}
        </button>
      </div>
    </header>
  );
}
```

- [ ] **Step 4: Create `MissionKpiStrip`**

Create `visual-workbench/web/src/components/MissionKpiStrip.tsx`:

```tsx
import type { MissionCockpitKpis } from "../types";

type MissionKpiStripProps = {
  kpis: MissionCockpitKpis;
};

export function MissionKpiStrip({ kpis }: MissionKpiStripProps) {
  return (
    <section className="mission-kpi-strip" aria-label="任务态势摘要">
      <KpiCard label="证据完整度" value={kpis.evidenceCompletenessLabel} />
      <KpiCard label="关键 artifact" value={String(kpis.keyArtifactCount)} />
      <KpiCard label="回放帧" value={kpis.replayFrameLabel} />
      <KpiCard label="剩余风险" value={kpis.riskLabel} />
    </section>
  );
}

function KpiCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="mission-kpi-card">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}
```

- [ ] **Step 5: Wire header and KPI strip in App**

In `visual-workbench/web/src/App.tsx`, replace the inline `<header className="mission-topbar">...</header>` block with:

```tsx
<MissionStatusHeader
  stage={selectedStage}
  health={health}
  presentationMode={presentationMode}
  onTogglePresentationMode={() => setPresentationMode((value) => !value)}
/>

<MissionKpiStrip kpis={cockpitKpis} />
```

Add imports:

```tsx
import { MissionKpiStrip } from "./components/MissionKpiStrip";
import { MissionStatusHeader } from "./components/MissionStatusHeader";
import { buildMissionCockpitKpis } from "./domain/missionCockpit";
```

Add this memo after `evidenceStage`:

```tsx
const cockpitKpis = useMemo(
  () => buildMissionCockpitKpis(evidenceStage, artifacts, route),
  [artifacts, evidenceStage, route],
);
```

Remove unused imports from `App.tsx`:

```tsx
import { PanelLeft, Presentation, Signal } from "lucide-react";
import { getStageDescription, getStageLabel } from "./components/MissionStageRail";
```

- [ ] **Step 6: Add minimal styles**

Append to `visual-workbench/web/src/styles.css`:

```css
.mission-status-header {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 18px;
  align-items: start;
}

.cockpit-label {
  margin: 4px 0 0;
  color: #075985;
  font-size: 13px;
  font-weight: 900;
}

.mission-kpi-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.mission-kpi-card {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  background: var(--surface);
}

.mission-kpi-card strong {
  display: block;
  color: var(--ink);
  font-size: 24px;
  line-height: 1.1;
}

.mission-kpi-card span {
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
}

@media (max-width: 760px) {
  .mission-status-header,
  .mission-kpi-strip {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 7: Run test and build**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
npm run build
```

Expected: App tests pass and build exits 0.

- [ ] **Step 8: Commit**

Run:

```powershell
git add visual-workbench/web/src/components/MissionStatusHeader.tsx visual-workbench/web/src/components/MissionKpiStrip.tsx visual-workbench/web/src/App.tsx visual-workbench/web/src/App.test.tsx visual-workbench/web/src/styles.css
git commit -m "feat: add h1 mission status and kpi strip"
```

---

### Task 3: Replace Map Panel With Interactive Mission Map Replay

**Files:**
- Create: `visual-workbench/web/src/components/MissionMapReplay.tsx`
- Modify: `visual-workbench/web/src/App.tsx`
- Modify: `visual-workbench/web/src/App.test.tsx`
- Modify: `visual-workbench/web/src/styles.css`

- [ ] **Step 1: Add failing map replay test**

Append this test inside `describe("App", ...)`:

```tsx
test("supports H1 map layer and timeline interactions", async () => {
  const user = userEvent.setup();
  render(<App />);

  expect(await screen.findByRole("region", { name: "地图回放主交互" })).toBeInTheDocument();
  const optimizedLayer = screen.getByRole("button", { name: "切换 optimized path 图层" });
  expect(optimizedLayer).toHaveAttribute("aria-pressed", "true");

  await user.click(optimizedLayer);

  expect(optimizedLayer).toHaveAttribute("aria-pressed", "false");

  const finalFrame = await screen.findByRole("button", { name: "选择回放帧 t2 目标接近" });
  await user.click(finalFrame);

  expect(finalFrame).toHaveAttribute("aria-current", "step");
});
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: FAIL because map replay region, layer buttons, and timeline buttons do not exist.

- [ ] **Step 3: Create `MissionMapReplay` by copying current drawing behavior**

Create `visual-workbench/web/src/components/MissionMapReplay.tsx`.

Start from the current `MissionMap.tsx`, then change the public component signature and render controls:

```tsx
import { useEffect, useRef, useState } from "react";

import { buildReplayFrames } from "../domain/missionCockpit";
import type { DerivedMissionStage } from "../domain/missionStages";
import type { MapLayerState, ReplayFrame, RoutePayload, SelectedMapObject, SidecarPayload } from "../types";
import { getStageLabel } from "./MissionStageRail";

type MissionMapReplayProps = {
  stage: DerivedMissionStage;
  sidecar: SidecarPayload | null;
  route: RoutePayload | null;
  selected: SelectedMapObject;
  layers: MapLayerState;
  onLayersChange: (layers: MapLayerState) => void;
  onSelectObject: (selection: SelectedMapObject) => void;
};
```

Use this render shell at the top of the component:

```tsx
export function MissionMapReplay({
  stage,
  sidecar,
  route,
  selected,
  layers,
  onLayersChange,
  onSelectObject,
}: MissionMapReplayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frames = buildReplayFrames(sidecar, route);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    drawMissionMap(ctx, canvas.width, canvas.height, sidecar, route, layers);
  }, [sidecar, route, layers]);

  return (
    <section className="mission-map-panel mission-map-replay" aria-label="地图回放主交互">
      <div className="map-panel-header">
        <div>
          <p className="section-label">Mission Map Replay</p>
          <h2>{getStageLabel(stage.id)} / 月面巡视状态</h2>
        </div>
        <span>{route?.reachable === false ? "路线需复核" : "交互回放"}</span>
      </div>

      <LayerControls layers={layers} onLayersChange={onLayersChange} />

      <button
        className="map-canvas-wrap map-canvas-button"
        type="button"
        aria-label={`选择地图对象 ${selected.label}`}
        onClick={() => onSelectObject(selected)}
      >
        <canvas ref={canvasRef} width={960} height={560} aria-label="月面任务地图" />
      </button>

      <ReplayTimeline frames={frames} selected={selected} onSelectObject={onSelectObject} />
    </section>
  );
}
```

Add these helper controls in the same file:

```tsx
function LayerControls({
  layers,
  onLayersChange,
}: {
  layers: MapLayerState;
  onLayersChange: (layers: MapLayerState) => void;
}) {
  const controls: Array<[keyof MapLayerState, string]> = [
    ["rawPath", "raw path"],
    ["smoothedPath", "smoothed path"],
    ["optimizedPath", "optimized path"],
    ["blocked", "blocked/passability"],
  ];

  return (
    <div className="map-layer-controls" aria-label="地图图层控制">
      {controls.map(([key, label]) => (
        <button
          key={key}
          type="button"
          className={layers[key] ? "map-layer-button active" : "map-layer-button"}
          aria-pressed={layers[key]}
          aria-label={`切换 ${label} 图层`}
          onClick={() => onLayersChange({ ...layers, [key]: !layers[key] })}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function ReplayTimeline({
  frames,
  selected,
  onSelectObject,
}: {
  frames: ReplayFrame[];
  selected: SelectedMapObject;
  onSelectObject: (selection: SelectedMapObject) => void;
}) {
  return (
    <div className="replay-timeline" aria-label="路径回放时间轴">
      {frames.map((frame) => (
        <button
          key={frame.frameId}
          type="button"
          className={frame.frameId === selected.frameId ? "replay-frame active" : "replay-frame"}
          aria-current={frame.frameId === selected.frameId ? "step" : undefined}
          aria-label={`选择回放帧 ${frame.frameId} ${frame.label}`}
          onClick={() => onSelectObject(frame)}
        >
          <strong>{frame.frameId}</strong>
          <span>{frame.label}</span>
        </button>
      ))}
    </div>
  );
}
```

Copy the drawing helpers from current `MissionMap.tsx`, but update `drawMissionMap` signature:

```ts
function drawMissionMap(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  sidecar: SidecarPayload | null,
  route: RoutePayload | null,
  layers: MapLayerState,
) {
  // Keep current normalization, grid drawing, goal drawing, and non-finite guards.
  // Gate these existing drawPath calls:
  if (layers.rawPath) drawPath(ctx, normalizePath(route?.geometric_path), cellWidth, cellHeight, "#f97316", 5);
  if (layers.smoothedPath) drawPath(ctx, normalizePath(route?.postprocess?.smoothed_path ?? route?.smoothed_path), cellWidth, cellHeight, "#06b6d4", 4);
  if (layers.optimizedPath) {
    const optimizedPath = normalizePath(route?.trajectory_optimization_report?.optimized_path);
    drawPath(
      ctx,
      optimizedPath.length ? optimizedPath : normalizePath(route?.trajectory_optimization_report?.resampled_optimized_path),
      cellWidth,
      cellHeight,
      "#22c55e",
      4,
    );
  }
}
```

When drawing cells, use `layers.blocked` to decide whether blocked passability is shown:

```ts
const blocked = layers.blocked && isBlocked(passableMask, row, col);
```

- [ ] **Step 4: Wire map replay in App**

In `App.tsx`, replace `MissionMap` import with:

```tsx
import { MissionMapReplay } from "./components/MissionMapReplay";
import { DEFAULT_MAP_LAYERS, firstReplaySelection } from "./domain/missionCockpit";
import type { MapLayerState, SelectedMapObject } from "./types";
```

Add state:

```tsx
const [mapLayers, setMapLayers] = useState<MapLayerState>(DEFAULT_MAP_LAYERS);
const [selectedMapObject, setSelectedMapObject] = useState<SelectedMapObject>(() => ({
  objectType: "mission-stage",
  frameId: "t0",
  label: "任务阶段",
}));
```

Add an effect after sidecar/route loading:

```tsx
useEffect(() => {
  setSelectedMapObject(firstReplaySelection(sidecar, route));
}, [sidecar, route]);
```

Replace:

```tsx
<MissionMap stage={selectedStage} sidecar={sidecar} route={route} />
```

with:

```tsx
<MissionMapReplay
  stage={selectedStage}
  sidecar={sidecar}
  route={route}
  selected={selectedMapObject}
  layers={mapLayers}
  onLayersChange={setMapLayers}
  onSelectObject={setSelectedMapObject}
/>
```

- [ ] **Step 5: Add map replay styles**

Append:

```css
.mission-map-replay {
  gap: 12px;
}

.map-layer-controls,
.replay-timeline {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.map-layer-button,
.replay-frame {
  min-height: 36px;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 7px 10px;
  background: #0f172a;
  color: #dbeafe;
  font-weight: 800;
  cursor: pointer;
}

.map-layer-button.active,
.replay-frame.active {
  border-color: #38bdf8;
  background: #075985;
  color: #f8fafc;
}

.map-canvas-button {
  width: 100%;
  padding: 0;
  text-align: inherit;
  cursor: pointer;
}

.replay-frame {
  display: grid;
  min-width: 92px;
  text-align: left;
}

.replay-frame span {
  color: #bfdbfe;
  font-size: 12px;
}
```

- [ ] **Step 6: Run tests and build**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
npm run build
```

Expected: App tests pass and build exits 0.

- [ ] **Step 7: Commit**

Run:

```powershell
git add visual-workbench/web/src/components/MissionMapReplay.tsx visual-workbench/web/src/App.tsx visual-workbench/web/src/App.test.tsx visual-workbench/web/src/styles.css
git commit -m "feat: add interactive mission map replay"
```

---

### Task 4: Add Bottom Evidence Chain Drawer

**Files:**
- Create: `visual-workbench/web/src/components/EvidenceChainDrawer.tsx`
- Modify: `visual-workbench/web/src/App.tsx`
- Modify: `visual-workbench/web/src/App.test.tsx`
- Modify: `visual-workbench/web/src/styles.css`

- [ ] **Step 1: Add failing drawer test**

Append:

```tsx
test("updates the evidence chain drawer from replay selection", async () => {
  const user = userEvent.setup();
  render(<App />);

  expect(await screen.findByRole("region", { name: "证据链抽屉" })).toBeInTheDocument();

  await user.click(await screen.findByRole("button", { name: "选择回放帧 t2 目标接近" }));

  const drawer = screen.getByRole("region", { name: "证据链抽屉" });
  expect(within(drawer).getByText("当前选中对象")).toBeInTheDocument();
  expect(within(drawer).getByText("目标接近")).toBeInTheDocument();
  expect(within(drawer).getByText("path-planner-route/v1")).toBeInTheDocument();
  expect(within(drawer).getByText("path-feedback-summary/v1")).toBeInTheDocument();
  expect(within(drawer).getByText(/禁止：full run/)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: FAIL because `证据链抽屉` is not rendered.

- [ ] **Step 3: Create `EvidenceChainDrawer`**

Create `visual-workbench/web/src/components/EvidenceChainDrawer.tsx`:

```tsx
import type { EvidenceChain } from "../types";

type EvidenceChainDrawerProps = {
  chain: EvidenceChain;
  presentationMode: boolean;
};

export function EvidenceChainDrawer({ chain, presentationMode }: EvidenceChainDrawerProps) {
  const visibleArtifacts = presentationMode ? chain.supportingArtifacts.slice(0, 3) : chain.supportingArtifacts;

  return (
    <section className="evidence-chain-drawer" aria-label="证据链抽屉">
      <div className="evidence-chain-section">
        <h2>当前选中对象</h2>
        <p>{chain.selected.label}</p>
        <code>{chain.selected.frameId}</code>
      </div>

      <div className="evidence-chain-section">
        <h2>Schema Flow</h2>
        <ol className="schema-flow">
          {chain.schemaFlow.map((item) => (
            <li key={item.schema} className={`schema-node ${item.status}`}>
              <span>{item.schema}</span>
              <strong>{item.status === "present" ? "present" : "missing"}</strong>
            </li>
          ))}
        </ol>
      </div>

      <div className="evidence-chain-section">
        <h2>Artifact 支撑</h2>
        {visibleArtifacts.length ? (
          <ul className="drawer-artifact-list">
            {visibleArtifacts.map((artifact) => (
              <li key={artifact.artifact_id}>
                <strong>{artifact.schema_version ?? artifact.kind}</strong>
                <span>{artifact.status}</span>
                {!presentationMode ? <code>{artifact.relative_path}</code> : null}
              </li>
            ))}
          </ul>
        ) : (
          <p>当前选中对象尚未关联 artifact。</p>
        )}
        <p className="drawer-action">{chain.nextSafeAction}</p>
        <p className="drawer-guard">禁止：{chain.forbiddenActions.join(" / ")}</p>
      </div>
    </section>
  );
}
```

- [ ] **Step 4: Wire drawer in App**

In `App.tsx`, add imports:

```tsx
import { EvidenceChainDrawer } from "./components/EvidenceChainDrawer";
import { buildEvidenceChain } from "./domain/missionCockpit";
```

Add memo:

```tsx
const evidenceChain = useMemo(
  () => buildEvidenceChain(evidenceStage, artifacts, selectedMapObject),
  [artifacts, evidenceStage, selectedMapObject],
);
```

Render after the main two-column `mission-shell`:

```tsx
<EvidenceChainDrawer chain={evidenceChain} presentationMode={presentationMode} />
```

Place it inside `main.mission-main`, below `mission-shell`, so it behaves as the H1 bottom drawer and does not stretch the map panel height.

- [ ] **Step 5: Add drawer styles**

Append:

```css
.evidence-chain-drawer {
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr) minmax(260px, 0.8fr);
  gap: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
  background: var(--surface);
}

.evidence-chain-section {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  background: var(--surface-soft);
}

.evidence-chain-section h2 {
  margin: 0 0 8px;
  font-size: 14px;
}

.schema-flow,
.drawer-artifact-list {
  display: grid;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.schema-node,
.drawer-artifact-list li {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
  align-items: center;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px;
  background: #ffffff;
}

.schema-node.missing {
  border-color: #fde68a;
  background: #fffbeb;
}

.schema-node span,
.drawer-artifact-list code {
  overflow-wrap: anywhere;
}

.drawer-action {
  margin: 10px 0 0;
  color: #075985;
  font-weight: 800;
}

.drawer-guard {
  margin: 8px 0 0;
  color: #991b1b;
  font-weight: 800;
}

@media (max-width: 1100px) {
  .evidence-chain-drawer {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 6: Run tests and build**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
npm run build
```

Expected: tests pass and build exits 0.

- [ ] **Step 7: Commit**

Run:

```powershell
git add visual-workbench/web/src/components/EvidenceChainDrawer.tsx visual-workbench/web/src/App.tsx visual-workbench/web/src/App.test.tsx visual-workbench/web/src/styles.css
git commit -m "feat: add evidence chain drawer"
```

---

### Task 5: Split And Strengthen Stage Tool Panels

**Files:**
- Create: `visual-workbench/web/src/components/StageToolPanel.tsx`
- Modify: `visual-workbench/web/src/App.tsx`
- Modify: `visual-workbench/web/src/App.test.tsx`
- Modify: `visual-workbench/web/src/styles.css`

- [ ] **Step 1: Add failing panel content tests**

Append:

```tsx
test("stage tools expose H1-specific evidence, replay, and validate panels", async () => {
  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByRole("button", { name: "Evidence Trace" }));
  expect(screen.getByRole("heading", { name: "Evidence Trace" })).toBeInTheDocument();
  expect(screen.getByText(/schema coverage/i)).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "Map Replay" }));
  expect(screen.getByRole("heading", { name: "Map Replay" })).toBeInTheDocument();
  expect(screen.getByText(/图层|时间轴/)).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "Validate" }));
  expect(screen.getByRole("heading", { name: "Validate" })).toBeInTheDocument();
  expect(screen.getByText(/dry-run 或 validate/)).toBeInTheDocument();
  expect(screen.getByText(/full run|完整 run/)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: FAIL because the current inline `StageToolPanel` does not expose H1-specific panel copy.

- [ ] **Step 3: Create `StageToolPanel`**

Create `visual-workbench/web/src/components/StageToolPanel.tsx`:

```tsx
import type { DerivedMissionStage } from "../domain/missionStages";
import type { EvidenceChain, ProjectStatus } from "../types";
import type { StageToolId } from "./StageTools";

type StageToolPanelProps = {
  toolId: StageToolId;
  stage: DerivedMissionStage;
  status: ProjectStatus | null;
  chain: EvidenceChain;
};

export function StageToolPanel({ toolId, stage, status, chain }: StageToolPanelProps) {
  return (
    <section className="stage-tool-panel" aria-label="阶段工具面板">
      {toolId === "evidence-trace" ? <EvidenceTracePanel stage={stage} chain={chain} /> : null}
      {toolId === "map-replay" ? <MapReplayPanel status={status} chain={chain} /> : null}
      {toolId === "validate" ? <ValidatePanel /> : null}
    </section>
  );
}

function EvidenceTracePanel({ stage, chain }: { stage: DerivedMissionStage; chain: EvidenceChain }) {
  return (
    <div className="legacy-tool-view">
      <h3>Evidence Trace</h3>
      <p>schema coverage：{chain.schemaFlow.filter((item) => item.status === "present").length}/{chain.schemaFlow.length}</p>
      <p>当前阶段关联 {stage.artifacts.length} 个 artifact；证据链抽屉显示缺失 schema 与禁止动作。</p>
    </div>
  );
}

function MapReplayPanel({ status, chain }: { status: ProjectStatus | null; chain: EvidenceChain }) {
  return (
    <div className="legacy-tool-view">
      <h3>Map Replay</h3>
      <p>地图回放可通过图层与时间轴定位当前对象：{chain.selected.label}。</p>
      <p>Repo root: {status?.repo_root ?? "加载中"}</p>
    </div>
  );
}

function ValidatePanel() {
  return (
    <div className="legacy-tool-view">
      <h3>Validate</h3>
      <p>Validate 入口只允许 dry-run 或 validate。</p>
      <p>完整 run / full run 不会从前端触发，也不得进入 PPO、training 或 staged release。</p>
    </div>
  );
}
```

- [ ] **Step 4: Refactor App to use `StageToolPanel`**

In `App.tsx`, remove:

```tsx
import { EvidenceTraceView, ValidateView } from "./components/LegacyWorkbenchViews";
```

Add:

```tsx
import { StageToolPanel } from "./components/StageToolPanel";
```

Delete the inline `StageToolPanel` function from the bottom of `App.tsx`.

Replace the render call:

```tsx
{activeTool ? <StageToolPanel toolId={activeTool} stage={evidenceStage} status={status} /> : null}
```

with:

```tsx
{activeTool ? (
  <StageToolPanel toolId={activeTool} stage={evidenceStage} status={status} chain={evidenceChain} />
) : null}
```

- [ ] **Step 5: Run tests and build**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
npm run build
```

Expected: tests pass and build exits 0.

- [ ] **Step 6: Commit**

Run:

```powershell
git add visual-workbench/web/src/components/StageToolPanel.tsx visual-workbench/web/src/App.tsx visual-workbench/web/src/App.test.tsx visual-workbench/web/src/styles.css
git commit -m "refactor: split h1 stage tool panels"
```

---

### Task 6: Finalize H1 Cockpit App Layout

**Files:**
- Modify: `visual-workbench/web/src/App.tsx`
- Modify: `visual-workbench/web/src/App.test.tsx`
- Modify: `visual-workbench/web/src/styles.css`

- [ ] **Step 1: Add failing layout contract test**

Append:

```tsx
test("renders the complete H1 cockpit structure", async () => {
  const { container } = render(<App />);

  expect(await screen.findByLabelText("任务阶段导航")).toBeInTheDocument();
  expect(screen.getByLabelText("任务态势摘要")).toBeInTheDocument();
  expect(screen.getByRole("region", { name: "地图回放主交互" })).toBeInTheDocument();
  expect(screen.getByRole("region", { name: "证据链抽屉" })).toBeInTheDocument();
  expect(container.querySelector(".h1-cockpit-main")).toBeInTheDocument();
  expect(container.querySelector(".mission-shell > .mission-map-stack + .mission-evidence-panel")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: FAIL until `App.tsx` uses the `h1-cockpit-main` class and complete H1 structure.

- [ ] **Step 3: Adjust App layout**

In `App.tsx`, change:

```tsx
<main className="mission-main">
```

to:

```tsx
<main className="mission-main h1-cockpit-main">
```

Ensure order inside `main` is:

```tsx
<MissionStatusHeader ... />
<MissionKpiStrip ... />
{loadError ? <ErrorState ... /> : null}
<section className="mission-shell" aria-label="任务控制台">
  <div className="mission-map-stack">
    <MissionMapReplay ... />
    <StageTools ... />
    {activeTool ? <StageToolPanel ... /> : null}
  </div>
  <MissionEvidencePanel ... />
</section>
<EvidenceChainDrawer ... />
```

In `selectStage`, reset selected map object and tool:

```tsx
function selectStage(stageId: MissionStageId) {
  userSelectedStage.current = true;
  setSelectedStageId(stageId);
  setActiveTool(null);
  setSelectedMapObject({
    objectType: "mission-stage",
    frameId: "t0",
    label: "任务阶段",
  });
}
```

- [ ] **Step 4: Stabilize layout CSS**

Append:

```css
.h1-cockpit-main {
  display: grid;
  gap: 16px;
  min-width: 0;
}

.h1-cockpit-main .mission-shell {
  align-items: start;
}

.mission-map-stack {
  align-self: start;
}
```

- [ ] **Step 5: Run app tests and build**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
npm run build
```

Expected: tests pass and build exits 0.

- [ ] **Step 6: Commit**

Run:

```powershell
git add visual-workbench/web/src/App.tsx visual-workbench/web/src/App.test.tsx visual-workbench/web/src/styles.css
git commit -m "feat: compose h1 cockpit layout"
```

---

### Task 7: Responsive And Accessibility Browser Smoke Check

**Files:**
- Modify only if needed: `visual-workbench/web/src/styles.css`
- Modify only if needed: affected component files with missing ARIA labels

- [ ] **Step 1: Run frontend unit tests first**

Run:

```powershell
cd visual-workbench\web
npm test -- --run
npm run build
```

Expected: all frontend tests pass and build exits 0 before browser smoke work begins.

- [ ] **Step 2: Start backend and frontend if not already running**

PowerShell backend:

```powershell
cd C:\Users\77634\.codex\worktrees\ca49\lunar-path-planning\visual-workbench
$env:VISUAL_WORKBENCH_ARTIFACT_ROOTS='C:\Users\77634\.codex\worktrees\ca49\lunar-path-planning\visual-workbench\fixtures\demo-artifacts'
python -m uvicorn visual_workbench.api:app --host 127.0.0.1 --port 8000
```

PowerShell frontend:

```powershell
cd C:\Users\77634\.codex\worktrees\ca49\lunar-path-planning\visual-workbench\web
npm run dev -- --host 127.0.0.1 --port 5173
```

- [ ] **Step 3: Run viewport smoke script**

Run from repo root:

```powershell
$env:NODE_PATH='D:\CodexDownloads\npm-cache\_npx\31e32ef8478fbf80\node_modules'
@'
const { chromium } = require('playwright-core');
const edgePath = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
(async () => {
  const browser = await chromium.launch({ executablePath: edgePath, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage();
  const results = [];
  for (const width of [375, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 950 });
    await page.goto('http://127.0.0.1:5173', { waitUntil: 'networkidle' });
    await page.keyboard.press('Tab');
    const result = await page.evaluate(() => {
      const doc = document.documentElement;
      const focused = document.activeElement;
      const style = focused ? getComputedStyle(focused) : null;
      return {
        width,
        title: document.querySelector('h1')?.textContent || '',
        overflowX: doc.scrollWidth > doc.clientWidth + 1 || document.body.scrollWidth > doc.clientWidth + 1,
        hasMapReplay: Boolean(document.querySelector('[aria-label="地图回放主交互"]')),
        hasDrawer: Boolean(document.querySelector('[aria-label="证据链抽屉"]')),
        hasCanvas: Boolean(document.querySelector('canvas[aria-label="月面任务地图"]')),
        focusedTag: focused?.tagName || '',
        focusOutline: style?.outlineStyle || '',
        focusOutlineWidth: style?.outlineWidth || '',
      };
    });
    results.push(result);
  }
  await browser.close();
  console.log(JSON.stringify(results, null, 2));
})();
'@ | node -
```

Expected for each viewport:

- `"overflowX": false`
- `"hasMapReplay": true`
- `"hasDrawer": true`
- `"hasCanvas": true`
- focus outline is visible, such as non-`none` style and non-`0px` width.

- [ ] **Step 4: Fix any responsive failures with scoped CSS**

If horizontal overflow appears, patch only the affected containers:

```css
.mission-main,
.mission-shell,
.mission-map-stack,
.mission-map-panel,
.mission-evidence-panel,
.evidence-chain-drawer,
.map-layer-controls,
.replay-timeline {
  min-width: 0;
}

.drawer-artifact-list code,
.schema-node span,
.raw-evidence code {
  overflow-wrap: anywhere;
}

@media (max-width: 760px) {
  .mission-shell,
  .mission-kpi-strip,
  .evidence-chain-drawer {
    grid-template-columns: 1fr;
  }

  .map-layer-button,
  .replay-frame,
  .stage-tool-button {
    width: 100%;
    justify-content: center;
  }
}
```

If focus outline is missing, restore global focus style:

```css
button:focus-visible,
[tabindex]:focus-visible {
  outline: 3px solid #38bdf8;
  outline-offset: 3px;
}
```

- [ ] **Step 5: Rerun smoke script**

Run the Step 3 script again.

Expected: all viewport entries pass the expected checks.

- [ ] **Step 6: Commit fixes if files changed**

If CSS or component files changed:

```powershell
git add visual-workbench/web/src/styles.css visual-workbench/web/src/components
git commit -m "fix: harden h1 cockpit responsive accessibility"
```

If no files changed, do not create an empty commit.

---

### Task 8: Update Documentation And Parent Boundary Test

**Files:**
- Modify: `visual-workbench/README.md`
- Modify: `tests/test_visual_workbench_boundary.py`
- Modify only if needed: `docs/superpowers/specs/2026-06-17-visual-workbench-h1-mission-cockpit-design.md`

- [ ] **Step 1: Add failing parent contract assertions**

In `tests/test_visual_workbench_boundary.py`, extend the visual-workbench documentation boundary test with these required strings:

```python
    for required in (
        "H1 mission cockpit",
        "mission-first interactive evidence cockpit",
        "MissionMapReplay",
        "EvidenceChainDrawer",
        "Evidence Trace",
        "Map Replay",
        "Validate",
        "dry-run",
        "full run",
        "PPO",
        "training",
    ):
        assert required in docs
```

If the existing test variable is named differently, use that existing README/spec text variable and keep the assertion strings unchanged.

- [ ] **Step 2: Run parent boundary test to verify failure**

Run:

```powershell
python -m pytest tests\test_visual_workbench_boundary.py
```

Expected: FAIL until README includes the H1 cockpit terms.

- [ ] **Step 3: Update README**

Add this section to `visual-workbench/README.md`:

```markdown
## H1 Mission Cockpit Direction

The next frontend direction is the H1 mission cockpit: a mission-first
interactive evidence cockpit for lunar survey status, map replay, evidence
traceability, and guarded validation.

The H1 first screen contains:

- left mission stage rail;
- top mission status and KPI strip;
- central MissionMapReplay with layer controls and a replay timeline;
- right judgment and tools panel with Evidence Trace, Map Replay, and Validate;
- bottom EvidenceChainDrawer for selected map objects, schema flow, missing
  evidence, safe next action, and forbidden actions.

Validate remains limited to dry-run and validate. The frontend must not trigger
full run, PPO, training, staged release, or changes to network/action
space/default A*.
```

- [ ] **Step 4: Run parent test again**

Run:

```powershell
python -m pytest tests\test_visual_workbench_boundary.py
```

Expected: visual-workbench boundary tests pass.

- [ ] **Step 5: Commit docs**

Run:

```powershell
git add visual-workbench/README.md tests/test_visual_workbench_boundary.py docs/superpowers/specs/2026-06-17-visual-workbench-h1-mission-cockpit-design.md
git commit -m "docs: document h1 mission cockpit boundary"
```

---

### Task 9: Final Verification

**Files:**
- No code changes unless verification exposes a real defect.

- [ ] **Step 1: Run backend tests**

Run:

```powershell
cd C:\Users\77634\.codex\worktrees\ca49\lunar-path-planning\visual-workbench
python -m pytest
```

Expected: all visual-workbench backend tests pass. Existing FastAPI/Starlette TestClient warnings are acceptable if tests pass.

- [ ] **Step 2: Run frontend tests**

Run:

```powershell
cd C:\Users\77634\.codex\worktrees\ca49\lunar-path-planning\visual-workbench\web
npm test -- --run
```

Expected: all Vitest tests pass.

- [ ] **Step 3: Run frontend build**

Run:

```powershell
cd C:\Users\77634\.codex\worktrees\ca49\lunar-path-planning\visual-workbench\web
npm run build
```

Expected: TypeScript and Vite build exit 0.

- [ ] **Step 4: Run parent contract tests**

Run:

```powershell
cd C:\Users\77634\.codex\worktrees\ca49\lunar-path-planning
python -m pytest tests\test_visual_workbench_boundary.py tests\test_path_planner_algorithm_report.py
```

Expected: both parent tests pass.

- [ ] **Step 5: Rerun viewport smoke**

Run the viewport smoke script from Task 7.

Expected:

- 375 / 768 / 1024 / 1440 widths have no horizontal overflow;
- map replay region exists;
- evidence-chain drawer exists;
- canvas exists;
- focus outline is visible.

- [ ] **Step 6: Record verification outcome**

If all checks pass, include these exact commands and pass counts in the final response:

```text
visual-workbench backend pytest: <count> passed
frontend Vitest: <count> passed
frontend build: passed
parent boundary/report pytest: <count> passed
viewport smoke: 375/768/1024/1440 passed
```

If an unrelated pre-existing audit failure appears outside the listed commands, do not broaden this task. Record the exact failing test and classify it as outside H1 cockpit scope.

## Final Completion Criteria

Implementation is complete when:

- H1 task cockpit is the first screen;
- left mission rail, top mission status, KPI strip, central interactive map replay, right judgment/tools, and bottom evidence-chain drawer all render;
- map layers can be toggled with `aria-pressed`;
- replay timeline selection updates selected map object and evidence-chain drawer;
- Evidence Trace, Map Replay, and Validate remain stage-local tools and toggle open/closed;
- Validate copy and behavior preserve dry-run / validate-only boundary;
- presentation mode hides low-level details without changing metric meaning;
- responsive smoke reports no horizontal overflow;
- all verification commands in Task 9 pass.
