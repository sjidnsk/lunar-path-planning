# Visual Workbench Mission-Control Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `visual-workbench` from a basic artifact browser into a mission-first lunar survey control room with A+B scope: a polished showcase MVP plus critical research entry points.

**Architecture:** Keep the existing FastAPI artifact API intact for the first slice. Add a frontend mission-stage domain model that derives stage state from `/api/artifacts`, then render a mission-first shell with dark map/status surface and light progressive evidence panel. Keep Evidence Trace, Map Replay, and Validate as stage-local entry points instead of top-level competing pages.

**Tech Stack:** React, TypeScript, Vite, Vitest, Testing Library, lucide-react, FastAPI, Pydantic, pytest.

---

## Scope Guard

This plan implements the confirmed redesign spec:

`docs/superpowers/specs/2026-06-17-visual-workbench-mission-control-redesign.md`

It does not introduce backend mission endpoints in this slice. The frontend derives the mission model from the existing artifact list and raw artifact endpoints.

This plan must preserve these boundaries:

- no PPO, training, staged release, or complete experiment `run`;
- no change to `network/action space/default A*`;
- no replacement of `dev-platform-constraints`, `model-explorer`, or `path-planner` responsibilities;
- no Ackermann-feasible trajectory claim;
- no promotion of IRIS/GCS/path-planner diagnostics into training approval, release proof, or performance proof.

## File Structure

Create focused frontend modules:

- `visual-workbench/web/src/types.ts`
  - Shared `Artifact`, `ProjectStatus`, `CommandResult`, sidecar, route, and artifact-detail types.
- `visual-workbench/web/src/api.ts`
  - Shared `getJson`, `fetchRawText`, `fetchRawJson`, and `postJson` helpers.
- `visual-workbench/web/src/domain/missionStages.ts`
  - Six mission stage definitions.
  - Artifact-to-stage grouping.
  - Stage status derivation.
  - Current-stage selection.
  - Panel summary derivation.
- `visual-workbench/web/src/domain/missionStages.test.ts`
  - Unit tests for stage grouping and status derivation.
- `visual-workbench/web/src/components/MissionStageRail.tsx`
  - Left rail and top stage strip.
- `visual-workbench/web/src/components/MissionEvidencePanel.tsx`
  - Progressive stage panel with Level 1 judgment, Level 2 credibility, and Level 3 raw evidence.
- `visual-workbench/web/src/components/MissionMap.tsx`
  - Mission map canvas using sidecar/route payloads.
- `visual-workbench/web/src/components/StageTools.tsx`
  - Stage-local Evidence Trace, Map Replay, and Validate entry points.
- `visual-workbench/web/src/components/LegacyWorkbenchViews.tsx`
  - Stage-local compatibility views for evidence and validation panels during the mission-shell migration.
- `visual-workbench/web/src/App.tsx`
  - Reduced to data loading, app mode state, selected stage state, and layout orchestration.
- `visual-workbench/web/src/styles.css`
  - Replace broad dashboard styling with mission shell tokens and responsive layouts.
- `visual-workbench/web/src/App.test.tsx`
  - End-to-end component tests for the redesigned shell.

Update documentation:

- `visual-workbench/README.md`
  - Add mission-control UI description and verification commands.
- `docs/superpowers/specs/2026-06-17-visual-workbench-mission-control-redesign.md`
  - Only update if implementation changes a design decision.

---

### Task 1: Add Shared Frontend Types and API Helpers

**Files:**
- Create: `visual-workbench/web/src/types.ts`
- Create: `visual-workbench/web/src/api.ts`
- Modify: `visual-workbench/web/src/App.tsx`
- Test: `visual-workbench/web/src/App.test.tsx`

- [ ] **Step 1: Write the failing import test**

Add this small smoke assertion to `visual-workbench/web/src/App.test.tsx` so the test suite fails until shared modules exist:

```tsx
import { describe, expect, test, vi, afterEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "./App";
import type { Artifact } from "./types";

test("shared Artifact type remains importable", () => {
  const artifact: Artifact = {
    artifact_id: "typed",
    name: "typed.json",
    relative_path: "outputs/typed.json",
    kind: "json",
    schema_version: "path-planner-route/v1",
    status: "passed",
    size_bytes: 42,
    modified_at: "2026-06-17T00:00:00Z",
  };

  expect(artifact.schema_version).toBe("path-planner-route/v1");
});
```

- [ ] **Step 2: Run the frontend tests to verify failure**

Run:

```powershell
cd visual-workbench\web
npm test -- --run
```

Expected: FAIL with a TypeScript/module resolution error for `./types`.

- [ ] **Step 3: Create shared type definitions**

Create `visual-workbench/web/src/types.ts`:

```ts
export type Artifact = {
  artifact_id: string;
  name: string;
  path?: string;
  relative_path: string;
  kind: string;
  schema_version: string | null;
  status: string;
  size_bytes: number;
  modified_at: string;
};

export type ProjectStatus = {
  repo_root: string;
  subprojects: Record<string, { exists: boolean; path?: string }>;
  artifact_roots?: string[];
};

export type CommandResult = {
  status: string;
  kind: string;
  action: string;
  return_code: number;
  stdout_json?: Record<string, unknown> | null;
  stderr?: string;
};

export type ArtifactDetailPayload = Artifact & {
  summary: Record<string, unknown>;
};

export type SidecarPayload = {
  schema_version?: string;
  grid?: { width?: number; height?: number };
  cost?: number[][];
  passable_mask?: boolean[][];
  top_goals?: Array<{ cell?: [number, number]; utility?: number; reachable?: boolean }>;
};

export type RoutePayload = {
  schema_version?: string;
  geometric_path?: Array<[number, number]>;
  smoothed_path?: Array<[number, number]>;
  postprocess?: { smoothed_path?: Array<[number, number]> };
  trajectory_optimization_report?: {
    optimized_path?: Array<[number, number]>;
    resampled_optimized_path?: Array<[number, number]>;
  };
  reachable?: boolean;
  path_cost?: number;
};
```

- [ ] **Step 4: Create API helpers**

Create `visual-workbench/web/src/api.ts`:

```ts
export async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchRawText(artifactId: string): Promise<string> {
  const response = await fetch(`/api/artifacts/${artifactId}/raw`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.text();
}

export async function fetchRawJson<T>(artifactId: string): Promise<T> {
  const raw = await fetchRawText(artifactId);
  return JSON.parse(raw) as T;
}

export async function postJson<T>(url: string, payload: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const parsed = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(String(parsed.detail ?? response.statusText));
  }
  return parsed as T;
}
```

- [ ] **Step 5: Refactor App imports without changing behavior**

In `visual-workbench/web/src/App.tsx`, remove local definitions for `Artifact`, `ProjectStatus`, `CommandResult`, `ArtifactDetailPayload`, `SidecarPayload`, `RoutePayload`, `getJson`, `fetchRawText`, `fetchRawJson`, and `postJson`.

Add:

```tsx
import { fetchRawJson, fetchRawText, getJson, postJson } from "./api";
import type {
  Artifact,
  ArtifactDetailPayload,
  CommandResult,
  ProjectStatus,
  RoutePayload,
  SidecarPayload,
} from "./types";
```

- [ ] **Step 6: Run tests and build**

Run:

```powershell
cd visual-workbench\web
npm test -- --run
npm run build
```

Expected: all frontend tests pass and Vite build exits 0.

- [ ] **Step 7: Commit**

```powershell
git add visual-workbench/web/src/types.ts visual-workbench/web/src/api.ts visual-workbench/web/src/App.tsx visual-workbench/web/src/App.test.tsx
git commit -m "refactor: share visual workbench frontend types and api helpers"
```

---

### Task 2: Add the Mission Stage Domain Model

**Files:**
- Create: `visual-workbench/web/src/domain/missionStages.ts`
- Create: `visual-workbench/web/src/domain/missionStages.test.ts`

- [ ] **Step 1: Write failing domain tests**

Create `visual-workbench/web/src/domain/missionStages.test.ts`:

```ts
import { describe, expect, test } from "vitest";
import type { Artifact } from "../types";
import { deriveMissionStages, MISSION_STAGES } from "./missionStages";

function artifact(schema_version: string, status = "passed", name = `${schema_version}.json`): Artifact {
  return {
    artifact_id: `${schema_version}-${status}`,
    name,
    relative_path: `outputs/${name}`,
    kind: "json",
    schema_version,
    status,
    size_bytes: 128,
    modified_at: "2026-06-17T00:00:00Z",
  };
}

describe("mission stage derivation", () => {
  test("defines six mission stages in approved narrative order", () => {
    expect(MISSION_STAGES.map((stage) => stage.label)).toEqual([
      "环境测绘",
      "目标捕获",
      "路线制导",
      "可达确认",
      "风险复核",
      "任务简报",
    ]);
  });

  test("groups artifacts into mission stages and selects route guidance as current", () => {
    const stages = deriveMissionStages([
      artifact("model-explorer-contract/v1"),
      artifact("path-planner-sidecar/v1"),
      artifact("path-planner-route/v1", "reachable", "route.json"),
    ]);

    expect(stages.find((stage) => stage.id === "environment-mapping")?.state).toBe("passed");
    expect(stages.find((stage) => stage.id === "route-guidance")?.state).toBe("current");
    expect(stages.find((stage) => stage.id === "reachability-confirmation")?.state).toBe("pending");
  });

  test("marks feedback validation as blocked when summary is blocked", () => {
    const stages = deriveMissionStages([
      artifact("path-planner-route/v1", "reachable", "route.json"),
      artifact("path-feedback-summary/v1", "blocked", "summary.json"),
    ]);

    expect(stages.find((stage) => stage.id === "reachability-confirmation")?.state).toBe("blocked");
    expect(stages.find((stage) => stage.id === "reachability-confirmation")?.risk).toContain("blocked");
  });
});
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/domain/missionStages.test.ts
```

Expected: FAIL because `./missionStages` does not exist.

- [ ] **Step 3: Implement mission stage derivation**

Create `visual-workbench/web/src/domain/missionStages.ts`:

```ts
import type { Artifact } from "../types";

export type MissionStageId =
  | "environment-mapping"
  | "target-capture"
  | "route-guidance"
  | "reachability-confirmation"
  | "risk-review"
  | "mission-briefing";

export type MissionStageState = "passed" | "current" | "pending" | "blocked";

export type MissionStageDefinition = {
  id: MissionStageId;
  label: string;
  description: string;
  schemas: string[];
};

export type DerivedMissionStage = MissionStageDefinition & {
  state: MissionStageState;
  artifacts: Artifact[];
  judgment: string;
  credibility: string;
  risk: string;
  nextAction: string;
};

export const MISSION_STAGES: MissionStageDefinition[] = [
  {
    id: "environment-mapping",
    label: "环境测绘",
    description: "读取地图、代价、可通行性和候选上下文。",
    schemas: ["model-explorer-contract/v1", "path-planner-sidecar/v1"],
  },
  {
    id: "target-capture",
    label: "目标捕获",
    description: "确认候选目标已经排序或被选中。",
    schemas: ["model-explorer-contract/v1", "model-explorer-experiment/v1"],
  },
  {
    id: "route-guidance",
    label: "路线制导",
    description: "生成可在地图上回放的候选路线。",
    schemas: ["path-planner-route/v1"],
  },
  {
    id: "reachability-confirmation",
    label: "可达确认",
    description: "通过反馈验证检查路线是否可达。",
    schemas: ["path-feedback-manifest/v1", "path-feedback-summary/v1"],
  },
  {
    id: "risk-review",
    label: "风险复核",
    description: "聚合失败原因、阻塞项和未完成边界。",
    schemas: ["path-feedback-summary/v1"],
  },
  {
    id: "mission-briefing",
    label: "任务简报",
    description: "把当前证据整理为可展示结论。",
    schemas: ["model-explorer-experiment/v1", "path-feedback-summary/v1"],
  },
];

const BLOCKED_STATUSES = new Set(["blocked", "failed", "error", "rejected"]);
const PASSED_STATUSES = new Set(["passed", "ready", "reachable"]);

export function deriveMissionStages(artifacts: Artifact[]): DerivedMissionStage[] {
  const grouped = MISSION_STAGES.map((stage) => {
    const stageArtifacts = artifacts.filter((artifact) => stage.schemas.includes(artifact.schema_version ?? ""));
    const hasBlocked = stageArtifacts.some((artifact) => BLOCKED_STATUSES.has(artifact.status));
    const hasPassed = stageArtifacts.some((artifact) => PASSED_STATUSES.has(artifact.status));
    const hasAny = stageArtifacts.length > 0;
    const baseState: MissionStageState = hasBlocked ? "blocked" : hasPassed || hasAny ? "passed" : "pending";
    return buildStage(stage, stageArtifacts, baseState);
  });

  const firstPendingIndex = grouped.findIndex((stage) => stage.state === "pending" || stage.state === "blocked");
  const currentIndex = firstPendingIndex === -1 ? grouped.length - 1 : Math.max(0, firstPendingIndex - 1);

  return grouped.map((stage, index) => {
    if (stage.state === "blocked") return stage;
    if (index === currentIndex) return { ...stage, state: "current" };
    return stage;
  });
}

export function getCurrentStage(stages: DerivedMissionStage[]): DerivedMissionStage {
  return stages.find((stage) => stage.state === "current") ?? stages[0];
}

function buildStage(
  stage: MissionStageDefinition,
  artifacts: Artifact[],
  state: MissionStageState
): DerivedMissionStage {
  const schemaList = artifacts.map((artifact) => artifact.schema_version).filter(Boolean).join(", ");
  const blocked = artifacts.find((artifact) => BLOCKED_STATUSES.has(artifact.status));
  const artifactCount = artifacts.length;
  return {
    ...stage,
    artifacts,
    state,
    judgment: judgmentFor(stage, state, artifactCount),
    credibility: artifactCount
      ? `已识别 ${artifactCount} 个相关 artifact：${schemaList}`
      : "尚未在允许的 artifact root 中发现对应证据。",
    risk: blocked
      ? `${blocked.name} reports ${blocked.status}.`
      : "当前阶段仍保持 artifact-first 展示；未完成验证不会被包装成发布证明。",
    nextAction: nextActionFor(stage.id, state),
  };
}

function judgmentFor(stage: MissionStageDefinition, state: MissionStageState, artifactCount: number): string {
  if (state === "blocked") return `${stage.label}存在阻塞，需要先查看失败证据。`;
  if (state === "pending") return `${stage.label}等待相关 artifact 进入 allowlist root。`;
  if (state === "current") return `${stage.label}是当前任务焦点。`;
  return artifactCount > 0 ? `${stage.label}已有证据支撑。` : `${stage.label}暂无证据。`;
}

function nextActionFor(stageId: MissionStageId, state: MissionStageState): string {
  if (state === "blocked") return "打开 Evidence Trace 查看 reason codes 和原始 artifact。";
  if (state === "pending") return "确认上游输出目录或配置 VISUAL_WORKBENCH_ARTIFACT_ROOTS。";
  if (stageId === "route-guidance") return "打开 Map Replay 回放路径，并进入可达确认。";
  if (stageId === "reachability-confirmation") return "使用 Validate 执行 dry-run 或 validate。";
  return "展开证据面板核对 artifact 摘要和原始内容。";
}
```

- [ ] **Step 4: Run domain tests**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/domain/missionStages.test.ts
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add visual-workbench/web/src/domain/missionStages.ts visual-workbench/web/src/domain/missionStages.test.ts
git commit -m "feat: derive mission stages from artifacts"
```

---

### Task 3: Build the Mission Stage Rail

**Files:**
- Create: `visual-workbench/web/src/components/MissionStageRail.tsx`
- Modify: `visual-workbench/web/src/App.test.tsx`
- Modify: `visual-workbench/web/src/App.tsx`

- [ ] **Step 1: Add failing UI test for the six-stage rail**

Append this test to `visual-workbench/web/src/App.test.tsx`:

```tsx
test("renders mission-first navigation with the approved stage names", async () => {
  render(<App />);

  await waitFor(() => expect(screen.getByText("环境测绘")).toBeInTheDocument());

  for (const label of ["环境测绘", "目标捕获", "路线制导", "可达确认", "风险复核", "任务简报"]) {
    expect(screen.getByRole("button", { name: label })).toBeInTheDocument();
  }

  expect(screen.queryByRole("button", { name: "Overview" })).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Run frontend test to verify failure**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: FAIL because mission stage buttons are not rendered yet.

- [ ] **Step 3: Create the stage rail component**

Create `visual-workbench/web/src/components/MissionStageRail.tsx`:

```tsx
import type { DerivedMissionStage, MissionStageId } from "../domain/missionStages";

type Props = {
  stages: DerivedMissionStage[];
  selectedStageId: MissionStageId;
  onSelectStage: (stageId: MissionStageId) => void;
};

export function MissionStageRail({ stages, selectedStageId, onSelectStage }: Props) {
  return (
    <aside className="mission-rail" aria-label="任务阶段导航">
      <div className="mission-brand">
        <span className="mission-mark" aria-hidden="true" />
        <div>
          <strong>Visual Workbench</strong>
          <span>Mission Control</span>
        </div>
      </div>
      <nav className="mission-nav">
        {stages.map((stage) => (
          <button
            key={stage.id}
            className={stage.id === selectedStageId ? `mission-nav-item active ${stage.state}` : `mission-nav-item ${stage.state}`}
            type="button"
            onClick={() => onSelectStage(stage.id)}
            aria-current={stage.id === selectedStageId ? "page" : undefined}
          >
            <span className="stage-state-dot" aria-hidden="true" />
            <span>{stage.label}</span>
            <small>{stateLabel(stage.state)}</small>
          </button>
        ))}
      </nav>
    </aside>
  );
}

export function MissionStageStrip({ stages, selectedStageId, onSelectStage }: Props) {
  return (
    <div className="mission-stage-strip" aria-label="任务阶段状态">
      {stages.map((stage) => (
        <button
          key={stage.id}
          className={stage.id === selectedStageId ? `stage-pill active ${stage.state}` : `stage-pill ${stage.state}`}
          type="button"
          onClick={() => onSelectStage(stage.id)}
        >
          <span>{stage.label}</span>
          <small>{stateLabel(stage.state)}</small>
        </button>
      ))}
    </div>
  );
}

function stateLabel(state: DerivedMissionStage["state"]) {
  if (state === "passed") return "已完成";
  if (state === "current") return "当前";
  if (state === "blocked") return "阻塞";
  return "等待";
}
```

- [ ] **Step 4: Wire the rail into App**

In `visual-workbench/web/src/App.tsx`, add:

```tsx
import { useEffect, useMemo, useState } from "react";
import { PanelLeft, Presentation } from "lucide-react";
import { getJson } from "./api";
import { deriveMissionStages, getCurrentStage, type MissionStageId } from "./domain/missionStages";
import { MissionStageRail, MissionStageStrip } from "./components/MissionStageRail";
import type { Artifact, ProjectStatus } from "./types";
```

Inside `App`, add:

```tsx
const missionStages = useMemo(() => deriveMissionStages(artifacts), [artifacts]);
const defaultStage = useMemo(() => getCurrentStage(missionStages), [missionStages]);
const [selectedStageId, setSelectedStageId] = useState<MissionStageId>("route-guidance");

useEffect(() => {
  setSelectedStageId((current) => (missionStages.some((stage) => stage.id === current) ? current : defaultStage.id));
}, [defaultStage.id, missionStages]);

const selectedStage = missionStages.find((stage) => stage.id === selectedStageId) ?? defaultStage;
```

Replace the existing `<aside className="sidebar">...</aside>` with:

```tsx
<MissionStageRail stages={missionStages} selectedStageId={selectedStage.id} onSelectStage={setSelectedStageId} />
```

Add the top strip inside `<main className="main">` after the header:

```tsx
<MissionStageStrip stages={missionStages} selectedStageId={selectedStage.id} onSelectStage={setSelectedStageId} />
```

Do not remove legacy view functions in this task; leave unused code cleanup for Task 8.

- [ ] **Step 5: Run the focused UI test**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: test for mission stage names passes; older tests may fail because legacy navigation names are gone. Update older tests only after Task 6 introduces the new shell behavior.

- [ ] **Step 6: Commit**

```powershell
git add visual-workbench/web/src/components/MissionStageRail.tsx visual-workbench/web/src/App.tsx visual-workbench/web/src/App.test.tsx
git commit -m "feat: add mission-first stage navigation"
```

---

### Task 4: Build the Progressive Evidence Panel

**Files:**
- Create: `visual-workbench/web/src/components/MissionEvidencePanel.tsx`
- Modify: `visual-workbench/web/src/App.test.tsx`
- Modify: `visual-workbench/web/src/App.tsx`

- [ ] **Step 1: Add failing test for progressive evidence**

Add to `visual-workbench/web/src/App.test.tsx`:

```tsx
test("shows mission judgment first and expands raw evidence on demand", async () => {
  const user = userEvent.setup();
  render(<App />);

  await waitFor(() => expect(screen.getByText("路线制导")).toBeInTheDocument());

  expect(screen.getByText(/当前任务焦点|已有证据支撑|等待相关 artifact/)).toBeInTheDocument();
  expect(screen.queryByText("Raw Evidence")).not.toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "展开证据" }));

  expect(await screen.findByText("Raw Evidence")).toBeInTheDocument();
  expect(screen.getByText(/path-planner-route\/v1|path-planner-sidecar\/v1/)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: FAIL because the progressive panel does not exist.

- [ ] **Step 3: Create evidence panel component**

Create `visual-workbench/web/src/components/MissionEvidencePanel.tsx`:

```tsx
import { ChevronDown, ChevronRight, FileJson, ShieldAlert } from "lucide-react";
import { useState } from "react";
import type { DerivedMissionStage } from "../domain/missionStages";

type Props = {
  stage: DerivedMissionStage;
  presentationMode: boolean;
};

export function MissionEvidencePanel({ stage, presentationMode }: Props) {
  const [expanded, setExpanded] = useState(false);
  const visibleArtifacts = presentationMode ? stage.artifacts.slice(0, 3) : stage.artifacts;

  return (
    <aside className="evidence-panel" aria-label="阶段证据面板">
      <div className="panel-kicker">当前阶段</div>
      <h2>{stage.label}</h2>
      <p className="stage-judgment">{stage.judgment}</p>

      <section className="evidence-section">
        <h3>为什么可信</h3>
        <p>{stage.credibility}</p>
      </section>

      <section className={stage.state === "blocked" ? "evidence-section risk blocked" : "evidence-section risk"}>
        <h3>
          <ShieldAlert size={16} aria-hidden="true" />
          未完成边界
        </h3>
        <p>{stage.risk}</p>
      </section>

      <section className="evidence-section">
        <h3>下一步</h3>
        <p>{stage.nextAction}</p>
      </section>

      <button
        className="button evidence-toggle"
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}
      >
        {expanded ? <ChevronDown size={16} aria-hidden="true" /> : <ChevronRight size={16} aria-hidden="true" />}
        展开证据
      </button>

      {expanded ? (
        <section className="raw-evidence">
          <h3>Raw Evidence</h3>
          {visibleArtifacts.length ? (
            <div className="artifact-trace-list">
              {visibleArtifacts.map((artifact) => (
                <article className="artifact-trace-row" key={artifact.artifact_id}>
                  <FileJson size={16} aria-hidden="true" />
                  <div>
                    <strong>{artifact.name}</strong>
                    <span>{artifact.schema_version ?? artifact.kind}</span>
                    {!presentationMode ? <small>{artifact.relative_path}</small> : null}
                  </div>
                  <span className={`trace-status ${artifact.status}`}>{artifact.status}</span>
                </article>
              ))}
            </div>
          ) : (
            <p>当前阶段尚未发现相关 artifact。</p>
          )}
        </section>
      ) : null}
    </aside>
  );
}
```

- [ ] **Step 4: Render panel from App**

In `visual-workbench/web/src/App.tsx`, import:

```tsx
import { MissionEvidencePanel } from "./components/MissionEvidencePanel";
```

Render it in the mission shell:

```tsx
<MissionEvidencePanel stage={selectedStage} presentationMode={presentationMode} />
```

Place it to the right of the dark map area in the new layout container introduced in Task 5. If Task 5 has not started, temporarily render it below the stage strip so the test can pass.

- [ ] **Step 5: Run test**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: progressive panel test passes.

- [ ] **Step 6: Commit**

```powershell
git add visual-workbench/web/src/components/MissionEvidencePanel.tsx visual-workbench/web/src/App.tsx visual-workbench/web/src/App.test.tsx
git commit -m "feat: add progressive mission evidence panel"
```

---

### Task 5: Build the Mission Map and Dark/Light Shell

**Files:**
- Create: `visual-workbench/web/src/components/MissionMap.tsx`
- Modify: `visual-workbench/web/src/App.tsx`
- Modify: `visual-workbench/web/src/styles.css`
- Test: `visual-workbench/web/src/App.test.tsx`

- [ ] **Step 1: Add failing test for map canvas and split shell**

Add to `visual-workbench/web/src/App.test.tsx`:

```tsx
test("renders the mission map canvas and split evidence shell", async () => {
  render(<App />);

  await waitFor(() => expect(screen.getByLabelText("月面任务地图")).toBeInTheDocument());

  expect(screen.getByLabelText("月面任务地图")).toBeInstanceOf(HTMLCanvasElement);
  expect(document.querySelector(".mission-shell")).not.toBeNull();
  expect(document.querySelector(".mission-map-surface")).not.toBeNull();
  expect(document.querySelector(".evidence-panel")).not.toBeNull();
});
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: FAIL because `.mission-shell` and `月面任务地图` do not exist.

- [ ] **Step 3: Create mission map component**

Create `visual-workbench/web/src/components/MissionMap.tsx`:

```tsx
import { useEffect, useRef } from "react";
import type { RoutePayload, SidecarPayload } from "../types";
import type { DerivedMissionStage } from "../domain/missionStages";

type Props = {
  stage: DerivedMissionStage;
  sidecar: SidecarPayload | null;
  route: RoutePayload | null;
};

export function MissionMap({ stage, sidecar, route }: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cost = sidecar?.cost ?? [
      [1, 1, 2, 4, 8],
      [1, 2, 3, 4, 8],
      [1, 1, 1, 2, 3],
      [4, 3, 2, 2, 1],
      [8, 4, 3, 1, 1],
    ];
    const height = cost.length;
    const width = cost[0]?.length ?? 1;
    const cell = Math.max(10, Math.floor(Math.min(canvas.width / width, canvas.height / height)));
    const max = Math.max(...cost.flat().filter(Number.isFinite), 1);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const blocked = sidecar?.passable_mask?.[y]?.[x] === false;
        const intensity = Math.min(1, (cost[y]?.[x] ?? 0) / max);
        ctx.fillStyle = blocked ? "#020617" : blend("#1e293b", "#94a3b8", intensity);
        ctx.fillRect(x * cell + 24, y * cell + 24, cell - 2, cell - 2);
      }
    }

    drawPath(ctx, route?.geometric_path ?? [], cell, "#fb923c", 4);
    drawPath(ctx, route?.postprocess?.smoothed_path ?? route?.smoothed_path ?? [], cell, "#38bdf8", 3);
    drawPath(ctx, route?.trajectory_optimization_report?.optimized_path ?? [], cell, "#22c55e", 3);

    ctx.fillStyle = "#dbeafe";
    ctx.font = "600 18px system-ui";
    ctx.fillText(stage.label, 24, canvas.height - 28);
  }, [sidecar, route, stage.label]);

  return (
    <section className="mission-map-surface" aria-label="任务地图区域">
      <div className="map-status-bar">
        <span>Mission Phase</span>
        <strong>{stage.label}</strong>
      </div>
      <div className="mission-canvas-frame">
        <canvas ref={ref} width={760} height={480} aria-label="月面任务地图" />
      </div>
      <div className="mission-map-legend" aria-label="地图图例">
        <span><i className="legend-swatch raw" />raw path</span>
        <span><i className="legend-swatch smooth" />smoothed path</span>
        <span><i className="legend-swatch optimized" />optimized path</span>
        <span><i className="legend-swatch blocked" />blocked cell</span>
      </div>
    </section>
  );
}

function drawPath(ctx: CanvasRenderingContext2D, path: Array<[number, number]>, cell: number, color: string, width: number) {
  if (path.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  path.forEach(([x, y], index) => {
    const px = x * cell + cell / 2 + 24;
    const py = y * cell + cell / 2 + 24;
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
}

function blend(from: string, to: string, amount: number) {
  const parse = (hex: string) => [1, 3, 5].map((start) => Number.parseInt(hex.slice(start, start + 2), 16));
  const [r1, g1, b1] = parse(from);
  const [r2, g2, b2] = parse(to);
  const mix = (a: number, b: number) => Math.round(a + (b - a) * amount);
  return `rgb(${mix(r1, r2)}, ${mix(g1, g2)}, ${mix(b1, b2)})`;
}
```

- [ ] **Step 4: Load default sidecar and route in App**

In `visual-workbench/web/src/App.tsx`, keep state:

```tsx
const [sidecar, setSidecar] = useState<SidecarPayload | null>(null);
const [route, setRoute] = useState<RoutePayload | null>(null);
```

Add an effect after artifacts load:

```tsx
useEffect(() => {
  const sidecarArtifact = artifacts.find((artifact) => artifact.schema_version === "path-planner-sidecar/v1");
  const routeArtifact = artifacts.find((artifact) => artifact.schema_version === "path-planner-route/v1");
  let cancelled = false;

  Promise.all([
    sidecarArtifact ? fetchRawJson<SidecarPayload>(sidecarArtifact.artifact_id) : Promise.resolve(null),
    routeArtifact ? fetchRawJson<RoutePayload>(routeArtifact.artifact_id) : Promise.resolve(null),
  ])
    .then(([sidecarPayload, routePayload]) => {
      if (!cancelled) {
        setSidecar(sidecarPayload);
        setRoute(routePayload);
      }
    })
    .catch(() => {
      if (!cancelled) {
        setSidecar(null);
        setRoute(null);
      }
    });

  return () => {
    cancelled = true;
  };
}, [artifacts]);
```

Render:

```tsx
<section className="mission-shell">
  <MissionMap stage={selectedStage} sidecar={sidecar} route={route} />
  <MissionEvidencePanel stage={selectedStage} presentationMode={presentationMode} />
</section>
```

- [ ] **Step 5: Replace shell styles**

Append to `visual-workbench/web/src/styles.css`:

```css
.mission-shell {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr);
  gap: 16px;
  align-items: stretch;
}

.mission-map-surface {
  min-width: 0;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 16px;
  background: #020617;
  color: #dbeafe;
}

.map-status-bar,
.mission-map-legend {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.mission-canvas-frame {
  margin: 14px 0;
  overflow: auto;
  border: 1px solid #334155;
  border-radius: 8px;
  background: #0f172a;
}

.mission-canvas-frame canvas {
  display: block;
  max-width: 100%;
  height: auto;
}

.mission-map-legend span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-height: 28px;
  font-size: 13px;
}

.legend-swatch {
  width: 18px;
  height: 4px;
  border-radius: 999px;
  background: #94a3b8;
}

.legend-swatch.raw { background: #fb923c; }
.legend-swatch.smooth { background: #38bdf8; }
.legend-swatch.optimized { background: #22c55e; }
.legend-swatch.blocked {
  height: 14px;
  background: #020617;
  border: 1px solid #94a3b8;
}

.evidence-panel {
  min-width: 0;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 16px;
  background: #ffffff;
  color: #0f172a;
}

@media (max-width: 1100px) {
  .mission-shell {
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

```powershell
git add visual-workbench/web/src/components/MissionMap.tsx visual-workbench/web/src/App.tsx visual-workbench/web/src/styles.css visual-workbench/web/src/App.test.tsx
git commit -m "feat: add mission map control-room shell"
```

---

### Task 6: Add Stage-Local Research Entrypoints

**Files:**
- Create: `visual-workbench/web/src/components/StageTools.tsx`
- Modify: `visual-workbench/web/src/App.tsx`
- Modify: `visual-workbench/web/src/App.test.tsx`

- [ ] **Step 1: Add failing test for research entrypoints**

Add to `visual-workbench/web/src/App.test.tsx`:

```tsx
test("shows stage-local research entry points without restoring tool-first navigation", async () => {
  render(<App />);

  await waitFor(() => expect(screen.getByText("Evidence Trace")).toBeInTheDocument());

  expect(screen.getByText("Map Replay")).toBeInTheDocument();
  expect(screen.getByText("Validate")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Evidence Browser" })).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: FAIL because stage-local tools do not exist.

- [ ] **Step 3: Create StageTools component**

Create `visual-workbench/web/src/components/StageTools.tsx`:

```tsx
import { FileSearch, Map, PlayCircle } from "lucide-react";
import type { DerivedMissionStage } from "../domain/missionStages";

type Props = {
  stage: DerivedMissionStage;
  onOpenTool: (tool: "evidence" | "map" | "validate") => void;
};

export function StageTools({ stage, onOpenTool }: Props) {
  return (
    <section className="stage-tools" aria-label={`${stage.label} 研发入口`}>
      <button className="stage-tool" type="button" onClick={() => onOpenTool("evidence")}>
        <FileSearch size={18} aria-hidden="true" />
        <span>Evidence Trace</span>
        <small>artifact / raw / schema</small>
      </button>
      <button className="stage-tool" type="button" onClick={() => onOpenTool("map")}>
        <Map size={18} aria-hidden="true" />
        <span>Map Replay</span>
        <small>layers / goals / overlay</small>
      </button>
      <button className="stage-tool" type="button" onClick={() => onOpenTool("validate")}>
        <PlayCircle size={18} aria-hidden="true" />
        <span>Validate</span>
        <small>dry-run / validate only</small>
      </button>
    </section>
  );
}
```

- [ ] **Step 4: Add tool panel state in App**

In `visual-workbench/web/src/App.tsx`:

```tsx
const [openTool, setOpenTool] = useState<"evidence" | "map" | "validate" | null>(null);
```

Render below `mission-shell`:

```tsx
<StageTools stage={selectedStage} onOpenTool={setOpenTool} />
{openTool ? <StageToolPanel tool={openTool} stage={selectedStage} onClose={() => setOpenTool(null)} /> : null}
```

Add a small local component at the bottom of `App.tsx` for this task:

```tsx
function StageToolPanel({
  tool,
  stage,
  onClose,
}: {
  tool: "evidence" | "map" | "validate";
  stage: DerivedMissionStage;
  onClose: () => void;
}) {
  const title = tool === "evidence" ? "Evidence Trace" : tool === "map" ? "Map Replay" : "Validate";
  return (
    <section className="stage-tool-panel">
      <div className="panel-heading">
        <h2>{title}</h2>
        <button className="button secondary" type="button" onClick={onClose}>
          关闭
        </button>
      </div>
      <p>
        {stage.label} 阶段入口。完整 run 不会从此处触发；Validate 只允许 dry-run 或 validate。
      </p>
    </section>
  );
}
```

Add these imports in `visual-workbench/web/src/App.tsx`:

```tsx
import { StageTools } from "./components/StageTools";
import type { DerivedMissionStage } from "./domain/missionStages";
```

- [ ] **Step 5: Style stage tools**

Append to `visual-workbench/web/src/styles.css`:

```css
.stage-tools {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.stage-tool {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 4px 10px;
  align-items: center;
  min-height: 76px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 12px;
  background: #ffffff;
  color: #0f172a;
  text-align: left;
}

.stage-tool small {
  grid-column: 2;
  color: #475569;
}

.stage-tool-panel {
  margin-top: 16px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 16px;
  background: #ffffff;
}

@media (max-width: 760px) {
  .stage-tools {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 6: Run tests**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: stage-local entrypoint test passes.

- [ ] **Step 7: Commit**

```powershell
git add visual-workbench/web/src/components/StageTools.tsx visual-workbench/web/src/App.tsx visual-workbench/web/src/styles.css visual-workbench/web/src/App.test.tsx
git commit -m "feat: add stage-local research entry points"
```

---

### Task 7: Preserve Validate Command Boundary in the New Shell

**Files:**
- Modify: `visual-workbench/web/src/components/StageTools.tsx`
- Modify: `visual-workbench/web/src/App.tsx`
- Modify: `visual-workbench/web/src/App.test.tsx`

- [ ] **Step 1: Add failing test for Validate boundary copy**

Add:

```tsx
test("validate entry point states dry-run and validate only", async () => {
  const user = userEvent.setup();
  render(<App />);

  await waitFor(() => expect(screen.getByText("Validate")).toBeInTheDocument());
  await user.click(screen.getByText("Validate"));

  expect(screen.getByText(/dry-run 或 validate/)).toBeInTheDocument();
  expect(screen.getByText(/完整 run 不会/)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify failure or missing copy**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: FAIL if the panel copy does not include both boundary phrases.

- [ ] **Step 3: Update Validate panel content**

In `StageToolPanel`, use this content:

```tsx
function StageToolPanel({
  tool,
  stage,
  onClose,
}: {
  tool: "evidence" | "map" | "validate";
  stage: DerivedMissionStage;
  onClose: () => void;
}) {
  const title = tool === "evidence" ? "Evidence Trace" : tool === "map" ? "Map Replay" : "Validate";
  const body =
    tool === "validate"
      ? "Validate 入口只允许 dry-run 或 validate。完整 run 不会从前端触发，也不会在后端进入 subprocess。"
      : tool === "evidence"
        ? "Evidence Trace 展示本阶段相关 artifact 的摘要、schema、状态和原始路径。"
        : "Map Replay 展示本阶段相关地图层、目标点、raw/smoothed/optimized path overlay。";

  return (
    <section className="stage-tool-panel">
      <div className="panel-heading">
        <h2>{title}</h2>
        <button className="button secondary" type="button" onClick={onClose}>
          关闭
        </button>
      </div>
      <p>{stage.label}：{body}</p>
    </section>
  );
}
```

- [ ] **Step 4: Run frontend tests**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: Validate boundary test passes.

- [ ] **Step 5: Commit**

```powershell
git add visual-workbench/web/src/App.tsx visual-workbench/web/src/App.test.tsx
git commit -m "test: preserve validate-only boundary in mission shell"
```

---

### Task 8: Clean Up Legacy Navigation and Update Tests

**Files:**
- Modify: `visual-workbench/web/src/App.tsx`
- Modify: `visual-workbench/web/src/App.test.tsx`
- Create: `visual-workbench/web/src/components/LegacyWorkbenchViews.tsx`

- [ ] **Step 1: Replace old tests with mission-shell tests**

Update `visual-workbench/web/src/App.test.tsx` so the main tests are:

```tsx
test("filters evidence through the progressive raw panel rather than tool-first navigation", async () => {
  const user = userEvent.setup();
  render(<App />);

  await waitFor(() => expect(screen.getByText("路线制导")).toBeInTheDocument());
  await user.click(screen.getByRole("button", { name: "展开证据" }));

  expect(await screen.findByText("Raw Evidence")).toBeInTheDocument();
  expect(screen.getByText(/path-feedback-summary\/v1|path-planner-route\/v1/)).toBeInTheDocument();
});

test("presentation mode hides low-level diagnostic emphasis", async () => {
  const user = userEvent.setup();
  render(<App />);

  await waitFor(() => expect(screen.getByText("Mission Control")).toBeInTheDocument());
  await user.click(screen.getByRole("button", { name: "切换到演示模式" }));

  expect(screen.getByText("演示模式")).toBeInTheDocument();
  expect(screen.queryByText(/stderr/i)).not.toBeInTheDocument();
});
```

Keep existing tests that still match the new product behavior. Remove tests that assert `Overview`, `Evidence Browser`, or `Map & Route` top-level navigation exists.

- [ ] **Step 2: Run tests to see remaining failures**

Run:

```powershell
cd visual-workbench\web
npm test -- --run src/App.test.tsx
```

Expected: FAIL only where App still renders removed legacy navigation or copied labels.

- [ ] **Step 3: Remove legacy top-level navigation state**

In `visual-workbench/web/src/App.tsx`, remove:

```tsx
const nav = [...]
type TabId = ...
const [active, setActive] = useState<TabId>("overview");
const [filter, setFilter] = useState("");
const filteredArtifacts = ...
```

Remove conditional rendering by `active`. The mission shell becomes the primary app body.

- [ ] **Step 4: Create stage-local compatibility views**

Create `visual-workbench/web/src/components/LegacyWorkbenchViews.tsx` with explicit stage-local panels. These panels preserve the research entry labels without restoring the old top-level navigation.

Use this export shape:

```tsx
import type { Artifact } from "../types";

export function EvidenceTraceView({ artifacts }: { artifacts: Artifact[] }) {
  return (
    <div className="legacy-view-note">
      Evidence Trace uses {artifacts.length} stage artifacts and raw evidence expansion.
    </div>
  );
}

export function ValidateView() {
  return <div className="legacy-view-note">Validate remains dry-run / validate only.</div>;
}
```

- [ ] **Step 5: Run frontend suite**

Run:

```powershell
cd visual-workbench\web
npm test -- --run
npm run build
```

Expected: all frontend tests pass and build exits 0.

- [ ] **Step 6: Commit**

```powershell
git add visual-workbench/web/src/App.tsx visual-workbench/web/src/App.test.tsx visual-workbench/web/src/components/LegacyWorkbenchViews.tsx
git commit -m "refactor: make mission shell the primary workbench interface"
```

---

### Task 9: Responsive Browser Smoke Check

**Files:**
- Modify only if the smoke check finds overflow: `visual-workbench/web/src/styles.css`

- [ ] **Step 1: Start backend and frontend with demo fixtures**

Run in PowerShell window 1:

```powershell
cd C:\Users\77634\.codex\worktrees\ca49\lunar-path-planning\visual-workbench
$env:VISUAL_WORKBENCH_ARTIFACT_ROOTS='C:\Users\77634\.codex\worktrees\ca49\lunar-path-planning\visual-workbench\fixtures\demo-artifacts'
python -m uvicorn visual_workbench.api:app --host 127.0.0.1 --port 8000
```

Run in PowerShell window 2:

```powershell
cd C:\Users\77634\.codex\worktrees\ca49\lunar-path-planning\visual-workbench\web
npm run dev -- --host 127.0.0.1 --port 5173
```

- [ ] **Step 2: Run Edge/Playwright viewport smoke script**

Use the same local Edge approach already proven in this worktree:

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
    await page.setViewportSize({ width, height: 900 });
    await page.goto('http://127.0.0.1:5173', { waitUntil: 'networkidle' });
    await page.keyboard.press('Tab');
    const result = await page.evaluate(() => {
      const doc = document.documentElement;
      const canvas = document.querySelector('canvas[aria-label="月面任务地图"]');
      const focused = document.activeElement;
      const style = focused ? getComputedStyle(focused) : null;
      return {
        title: document.querySelector('h1')?.textContent || '',
        width: doc.clientWidth,
        scrollWidth: doc.scrollWidth,
        bodyScrollWidth: document.body.scrollWidth,
        overflowX: doc.scrollWidth > doc.clientWidth + 1 || document.body.scrollWidth > doc.clientWidth + 1,
        hasCanvas: Boolean(canvas),
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

Expected:

- every item has `"overflowX": false`;
- every item has `"hasCanvas": true`;
- focus outline is visible, for example `"solid"` and non-zero width.

- [ ] **Step 3: Fix any overflow with scoped CSS**

If a viewport overflows, adjust only the overflowing container. Prefer:

```css
.mission-shell,
.mission-map-surface,
.evidence-panel,
.stage-tools {
  min-width: 0;
}

.evidence-panel pre,
.artifact-trace-row {
  overflow-wrap: anywhere;
}
```

Then rerun Step 2.

- [ ] **Step 4: Commit responsive fixes**

If CSS changed:

```powershell
git add visual-workbench/web/src/styles.css
git commit -m "fix: prevent mission shell overflow on small viewports"
```

If no CSS changed, do not create an empty commit.

---

### Task 10: Update Documentation and Parent Contract Test

**Files:**
- Modify: `visual-workbench/README.md`
- Modify: `tests/test_visual_workbench_boundary.py`

- [ ] **Step 1: Add failing parent contract assertions**

In `tests/test_visual_workbench_boundary.py`, extend `test_visual_workbench_structure_and_documented_boundaries`:

```python
    for required in (
        "mission-first",
        "环境测绘",
        "目标捕获",
        "路线制导",
        "可达确认",
        "风险复核",
        "任务简报",
        "progressive evidence",
        "Evidence Trace",
        "Map Replay",
        "Validate",
    ):
        assert required in docs
```

- [ ] **Step 2: Run parent test to verify failure**

Run:

```powershell
python -m pytest tests\test_visual_workbench_boundary.py
```

Expected: FAIL until README includes the mission-control redesign terms.

- [ ] **Step 3: Update README**

Add to `visual-workbench/README.md` after the scope section:

```markdown
## Mission-Control Direction

The next UI direction is mission-first rather than tool-first. The primary
navigation is:

1. 环境测绘
2. 目标捕获
3. 路线制导
4. 可达确认
5. 风险复核
6. 任务简报

The interface uses a split dark/light model: a dark mission map and status
surface paired with a light progressive evidence panel. Each stage keeps three
research entry points: Evidence Trace, Map Replay, and Validate. Validate remains
limited to dry-run and validate commands; full run is rejected.
```

- [ ] **Step 4: Run parent test**

Run:

```powershell
python -m pytest tests\test_visual_workbench_boundary.py
```

Expected: 2 tests pass.

- [ ] **Step 5: Run full verification set**

Run:

```powershell
cd visual-workbench
python -m pytest
cd web
npm test -- --run
npm run build
cd ..\..
python -m pytest tests\test_visual_workbench_boundary.py tests\test_path_planner_algorithm_report.py
```

Expected:

- backend tests pass;
- frontend tests pass;
- build passes;
- parent visual-workbench contract and algorithm report tests pass.

If `tests\test_project_code_docs_audit.py` is run and still fails on Global 99 findings, record the exact P0/P1 findings as external to this redesign unless this task explicitly expands to Global 99 evidence repair.

- [ ] **Step 6: Commit**

```powershell
git add visual-workbench/README.md tests/test_visual_workbench_boundary.py
git commit -m "docs: document mission-control visual workbench direction"
```

---

## Final Verification Checklist

Run these commands after all tasks:

```powershell
cd C:\Users\77634\.codex\worktrees\ca49\lunar-path-planning\visual-workbench
python -m pytest
cd web
npm test -- --run
npm run build
cd ..\..
python -m pytest tests\test_visual_workbench_boundary.py tests\test_path_planner_algorithm_report.py
```

Expected:

- `visual-workbench` backend pytest: all tests pass;
- frontend Vitest: all tests pass;
- Vite build: exits 0;
- parent visual-workbench boundary/report tests: pass.

Also run the viewport smoke check from Task 9 at 375, 768, 1024, and 1440 widths.

## Completion Criteria

The implementation is complete when:

- first screen is mission-first, not a generic artifact dashboard;
- six approved mission stages render in navigation and status strip;
- dark mission map and light evidence panel are visible;
- evidence panel hides raw evidence by default and expands it on user action;
- Evidence Trace, Map Replay, and Validate appear as stage-local research entry points;
- Validate copy and behavior preserve dry-run/validate-only boundary;
- presentation mode does not rewrite metric meaning and hides low-level diagnostic emphasis;
- responsive smoke check reports no horizontal overflow;
- verification commands in the final checklist pass, except unrelated pre-existing Global 99 audit findings if that broader audit is run.
