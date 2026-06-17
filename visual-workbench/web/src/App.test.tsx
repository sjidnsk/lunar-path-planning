import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { App } from "./App";
import { MissionMapReplay } from "./components/MissionMapReplay";
import type { DerivedMissionStage } from "./domain/missionStages";
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

const artifacts: Artifact[] = [
  {
    artifact_id: "contract-1",
    name: "contract.json",
    schema_version: "model-explorer-contract/v1",
    kind: "json",
    status: "passed",
    relative_path: "outputs/contract.json",
    modified_at: "2026-06-17T00:00:00Z",
    size_bytes: 180,
  },
  {
    artifact_id: "sidecar-1",
    name: "sidecar.json",
    schema_version: "path-planner-sidecar/v1",
    kind: "json",
    status: "passed",
    relative_path: "outputs/sidecar.json",
    modified_at: "2026-06-17T00:00:00Z",
    size_bytes: 220,
  },
  {
    artifact_id: "route-1",
    name: "route.json",
    schema_version: "path-planner-route/v1",
    kind: "json",
    status: "reachable",
    relative_path: "outputs/route.json",
    modified_at: "2026-06-17T00:00:00Z",
    size_bytes: 240,
  },
  {
    artifact_id: "experiment-1",
    name: "experiment.json",
    schema_version: "model-explorer-experiment/v1",
    kind: "json",
    status: "passed",
    relative_path: "outputs/experiment.json",
    modified_at: "2026-06-17T00:00:00Z",
    size_bytes: 260,
  },
  {
    artifact_id: "feedback-1",
    name: "path-feedback-summary.json",
    schema_version: "path-feedback-summary/v1",
    kind: "json",
    status: "passed",
    relative_path: "outputs/path-feedback-summary.json",
    modified_at: "2026-06-17T00:00:00Z",
    size_bytes: 120,
  },
];

function malformedArtifacts(): Artifact[] {
  return [
    {
      artifact_id: "bad-sidecar",
      name: "bad-sidecar.json",
      schema_version: "path-planner-sidecar/v1",
      kind: "json",
      status: "passed",
      relative_path: "outputs/bad-sidecar.json",
      modified_at: "2026-06-17T00:00:00Z",
      size_bytes: 10,
    },
    {
      artifact_id: "bad-route",
      name: "bad-route.json",
      schema_version: "path-planner-route/v1",
      kind: "json",
      status: "reachable",
      relative_path: "outputs/bad-route.json",
      modified_at: "2026-06-17T00:00:00Z",
      size_bytes: 10,
    },
  ];
}

function routeGuidanceArtifacts(): Artifact[] {
  return artifacts.filter((artifact) => ["contract-1", "sidecar-1", "route-1"].includes(artifact.artifact_id));
}

function firstStageButton(label: string): HTMLElement {
  return screen.getAllByRole("button", { name: label })[0];
}

let canvasCalls: Array<{ method: string; values: number[] }> = [];

function installCanvasContextMock() {
  const calls: Array<{ method: string; values: number[] }> = [];
  const context = {
    beginPath: vi.fn(() => calls.push({ method: "beginPath", values: [] })),
    clearRect: vi.fn((...values: number[]) => calls.push({ method: "clearRect", values })),
    fill: vi.fn(() => calls.push({ method: "fill", values: [] })),
    fillRect: vi.fn((...values: number[]) => calls.push({ method: "fillRect", values })),
    lineTo: vi.fn((...values: number[]) => calls.push({ method: "lineTo", values })),
    moveTo: vi.fn((...values: number[]) => calls.push({ method: "moveTo", values })),
    restore: vi.fn(() => calls.push({ method: "restore", values: [] })),
    save: vi.fn(() => calls.push({ method: "save", values: [] })),
    stroke: vi.fn(() => calls.push({ method: "stroke", values: [] })),
    arc: vi.fn((...values: number[]) => calls.push({ method: "arc", values })),
    set fillStyle(_value: string) {},
    set strokeStyle(_value: string) {},
    set lineWidth(_value: number) {},
    set lineCap(_value: CanvasLineCap) {},
    set lineJoin(_value: CanvasLineJoin) {},
    set shadowColor(_value: string) {},
    set shadowBlur(_value: number) {},
  } as unknown as CanvasRenderingContext2D;

  const spy = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => context);
  return { calls, spy };
}

beforeEach(() => {
  canvasCalls = installCanvasContextMock().calls;
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/health")) {
        return Promise.resolve(new Response(JSON.stringify({ status: "ok" })));
      }
      if (url.endsWith("/api/project/status")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              repo_root: "repo",
              subprojects: {
                "dev-platform-constraints": { exists: true },
                "model-explorer": { exists: true },
                "path-planner": { exists: true },
              },
            }),
          ),
        );
      }
      if (url.endsWith("/api/artifacts")) {
        return Promise.resolve(new Response(JSON.stringify({ artifacts })));
      }
      if (url.endsWith("/api/artifacts/sidecar-1/raw")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              schema_version: "path-planner-sidecar/v1",
              cost: [
                [1, 2, 3],
                [2, 4, 8],
                [1, 2, 1],
              ],
              passable_mask: [
                [true, true, true],
                [true, false, true],
                [true, true, true],
              ],
              top_goals: [{ cell: [2, 2], utility: 0.92, reachable: true }],
            }),
          ),
        );
      }
      if (url.endsWith("/api/artifacts/route-1/raw")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              schema_version: "path-planner-route/v1",
              geometric_path: [
                [0, 0],
                [1, 1],
                [2, 2],
              ],
              postprocess: {
                smoothed_path: [
                  [0, 0],
                  [1, 0.7],
                  [2, 2],
                ],
              },
              trajectory_optimization_report: {
                optimized_path: [
                  [0, 0],
                  [1.2, 0.8],
                  [2, 2],
                ],
                resampled_optimized_path: [
                  [0, 0],
                  [0.8, 0.6],
                  [2, 2],
                ],
              },
            }),
          ),
        );
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("App", () => {
  test("renders the complete H1 cockpit structure", async () => {
    const { container } = render(<App />);

    expect(await screen.findByLabelText("任务阶段导航")).toBeInTheDocument();
    expect(screen.getByLabelText("任务态势摘要")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "地图回放主交互" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "证据链抽屉" })).toBeInTheDocument();
    expect(container.querySelector(".h1-cockpit-main")).toBeInTheDocument();
    expect(container.querySelector(".mission-shell > .mission-map-stack + .mission-evidence-panel")).toBeInTheDocument();
  });

  test("evidence chain drawer follows the selected replay object", async () => {
    const user = userEvent.setup();
    render(<App />);

    const drawer = within(await screen.findByRole("region", { name: "证据链抽屉" }));
    expect(drawer.getByText("Selected Map Object")).toBeInTheDocument();
    expect(await drawer.findByText(/目标接近/)).toBeInTheDocument();
    expect(drawer.getByText("Schema Flow")).toBeInTheDocument();
    expect(await drawer.findByText("path-planner-route/v1")).toBeInTheDocument();
    expect(drawer.getByText("path-feedback-summary/v1")).toBeInTheDocument();
    expect(drawer.getByText(/禁止：full run \/ PPO \/ training/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /t0/ }));
    expect(drawer.getByText(/当前对象：起点/)).toBeInTheDocument();
    expect(drawer.getByText(/起点/)).toBeInTheDocument();
  });

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

  test("follows the derived current stage after async artifacts load before manual selection", async () => {
    render(<App />);

    const evidencePanel = await screen.findByRole("region", { name: "任务简报证据" });
    expect(evidencePanel).toBeInTheDocument();
    expect(within(evidencePanel).getByText(/任务简报是当前任务焦点/)).toBeInTheDocument();
  });

  test("preserves manual stage selection made before async artifacts finish loading", async () => {
    const user = userEvent.setup();
    let resolveArtifacts!: (response: Response) => void;
    const artifactResponse = new Promise<Response>((resolve) => {
      resolveArtifacts = resolve;
    });
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/health")) {
        return Promise.resolve(new Response(JSON.stringify({ status: "ok" })));
      }
      if (url.endsWith("/api/project/status")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              repo_root: "repo",
              subprojects: {
                "dev-platform-constraints": { exists: true },
                "model-explorer": { exists: true },
                "path-planner": { exists: true },
              },
            }),
          ),
        );
      }
      if (url.endsWith("/api/artifacts")) {
        return artifactResponse;
      }
      if (url.endsWith("/api/artifacts/sidecar-1/raw")) {
        return Promise.resolve(new Response(JSON.stringify({ schema_version: "path-planner-sidecar/v1" })));
      }
      if (url.endsWith("/api/artifacts/route-1/raw")) {
        return Promise.resolve(new Response(JSON.stringify({ schema_version: "path-planner-route/v1" })));
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await waitFor(() => expect(screen.getAllByRole("button", { name: "任务简报" }).length).toBeGreaterThan(0));
    await user.click(firstStageButton("任务简报"));

    resolveArtifacts(new Response(JSON.stringify({ artifacts: routeGuidanceArtifacts() })));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/artifacts/route-1/raw"));
    const evidencePanel = screen.getByRole("region", { name: "任务简报证据" });
    expect(evidencePanel).toBeInTheDocument();
    expect(within(evidencePanel).getByText(/任务简报尚未形成足够证据/)).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "路线制导证据" })).not.toBeInTheDocument();
  });

  test("renders mission-first navigation with the approved stage names", async () => {
    const { container } = render(<App />);

    await waitFor(() => expect(screen.getAllByRole("button", { name: "环境测绘" }).length).toBeGreaterThan(0));

    for (const label of ["环境测绘", "目标捕获", "路线制导", "可达确认", "风险复核", "任务简报"]) {
      expect(screen.getAllByRole("button", { name: label })).toHaveLength(1);
    }
    expect(screen.getByLabelText("任务阶段导航")).toBeInTheDocument();
    expect(container.querySelector(".mission-stage-strip")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Overview" })).not.toBeInTheDocument();
  });

  test("keeps API status in a dedicated corner outside view controls", async () => {
    const { container } = render(<App />);

    expect(await screen.findByLabelText("API 状态：ok")).toHaveClass("connection-status");
    expect(container.querySelector(".mission-status-header > .connection-status")).toBeInTheDocument();
    expect(container.querySelector(".mission-status-actions .connection-status")).not.toBeInTheDocument();
  });

  test("shows mission judgment first and expands raw evidence on demand", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => expect(screen.getAllByRole("button", { name: "路线制导" }).length).toBeGreaterThan(0));
    await user.click(firstStageButton("路线制导"));

    const panel = screen.getByRole("region", { name: "路线制导证据" });
    expect(within(panel).getByText("阶段判断")).toBeInTheDocument();
    expect(within(panel).getByText("可信度")).toBeInTheDocument();
    expect(within(panel).queryByRole("heading", { name: "Raw Evidence" })).not.toBeInTheDocument();

    await user.click(within(panel).getByRole("button", { name: "查看证据链" }));

    expect(within(panel).getByRole("heading", { name: "Raw Evidence" })).toBeInTheDocument();
    expect(within(panel).getByText("path-planner-route/v1")).toBeInTheDocument();
    expect(within(panel).getByRole("button", { name: "隐藏证据链" })).toHaveAttribute("aria-expanded", "true");

    await user.click(within(panel).getByRole("button", { name: "隐藏证据链" }));

    expect(within(panel).queryByRole("heading", { name: "Raw Evidence" })).not.toBeInTheDocument();
    expect(within(panel).getByRole("button", { name: "查看证据链" })).toHaveAttribute("aria-expanded", "false");
  });

  test("shows stage-local research entry points without restoring tool-first navigation", async () => {
    const { container } = render(<App />);

    expect(await screen.findByRole("button", { name: "Evidence Trace" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Map Replay" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Validate" })).toBeInTheDocument();
    expect(container.querySelector(".mission-map-stack .stage-tools")).toBeInTheDocument();
    expect(container.querySelector(".mission-shell > .mission-map-stack + .mission-evidence-panel")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Evidence Browser" })).not.toBeInTheDocument();
  });

  test("stage tools expose H1-specific evidence, replay, and validate panels", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Evidence Trace" }));
    let stageToolPanel = within(screen.getByRole("region", { name: "阶段工具面板" }));
    expect(stageToolPanel.getByRole("heading", { name: "Evidence Trace" })).toBeInTheDocument();
    expect(stageToolPanel.getByText(/schema coverage/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Map Replay" }));
    stageToolPanel = within(screen.getByRole("region", { name: "阶段工具面板" }));
    expect(stageToolPanel.getByRole("heading", { name: "Map Replay" })).toBeInTheDocument();
    expect(stageToolPanel.getByText(/图层|时间轴/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Validate" }));
    stageToolPanel = within(screen.getByRole("region", { name: "阶段工具面板" }));
    expect(stageToolPanel.getByRole("heading", { name: "Validate" })).toBeInTheDocument();
    expect(stageToolPanel.getByText(/dry-run 或 validate/)).toBeInTheDocument();
    expect(stageToolPanel.getByText(/full run|完整 run/)).toBeInTheDocument();
  });

  test("stage selection keeps the mission-stage fallback selected and closes active tools", async () => {
    const user = userEvent.setup();
    render(<App />);

    const drawer = within(await screen.findByRole("region", { name: "证据链抽屉" }));
    const replay = within(await screen.findByRole("region", { name: "地图回放主交互" }));
    expect(await drawer.findByText(/当前对象：目标接近/)).toBeInTheDocument();
    expect(await replay.findByText(/当前对象：目标接近/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Evidence Trace" }));
    expect(screen.getByRole("region", { name: "阶段工具面板" })).toBeInTheDocument();

    await user.click(firstStageButton("环境测绘"));

    expect(screen.queryByRole("region", { name: "阶段工具面板" })).not.toBeInTheDocument();
    expect(drawer.getByText(/当前对象：任务阶段/)).toBeInTheDocument();
    expect(drawer.getByText("mission-stage")).toBeInTheDocument();
    expect(replay.getByText(/当前对象：任务阶段/)).toBeInTheDocument();
    expect(replay.queryByText(/当前对象：起点/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /t0/ })).toHaveAttribute("aria-pressed", "false");
  });

  test("toggles a stage-local tool panel closed when the active tool is clicked again", async () => {
    const user = userEvent.setup();
    render(<App />);

    const evidenceButton = await screen.findByRole("button", { name: "Evidence Trace" });
    expect(evidenceButton).toHaveAttribute("aria-expanded", "false");

    await user.click(evidenceButton);

    expect(evidenceButton).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("heading", { name: "Evidence Trace" })).toBeInTheDocument();

    await user.click(evidenceButton);

    expect(evidenceButton).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("heading", { name: "Evidence Trace" })).not.toBeInTheDocument();
  });

  test("validate entry point states dry-run and validate only", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Validate" }));

    expect(screen.getByText(/dry-run 或 validate/)).toBeInTheDocument();
    expect(screen.getByText(/完整 run 不会/)).toBeInTheDocument();
  });

  test("filters evidence through the progressive raw panel rather than tool-first navigation", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => expect(screen.getAllByRole("button", { name: "可达确认" }).length).toBeGreaterThan(0));
    await user.click(firstStageButton("可达确认"));
    expect(screen.queryByRole("button", { name: "Evidence Browser" })).not.toBeInTheDocument();

    const panel = screen.getByRole("region", { name: "可达确认证据" });
    await user.click(within(panel).getByRole("button", { name: "查看证据链" }));

    expect(within(panel).getByText("path-planner-route/v1")).toBeInTheDocument();
    expect(within(panel).getByText("path-feedback-summary/v1")).toBeInTheDocument();
  });

  test("presentation mode hides low-level diagnostic emphasis", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText("研发模式")).toHaveClass("research");

    await user.click(await screen.findByRole("button", { name: "切换到演示模式" }));

    expect(screen.getByText("演示模式")).toHaveClass("presentation");
    expect(screen.queryByText(/stderr/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "切换到研发模式" }));

    expect(screen.getByText("研发模式")).toHaveClass("research");
  });

  test("presentation mode hides evidence chain drawer schema and path details", async () => {
    const user = userEvent.setup();
    render(<App />);

    const drawerRegion = await screen.findByRole("region", { name: "证据链抽屉" });
    const drawer = within(drawerRegion);
    expect(await drawer.findByText("path-planner-route/v1")).toBeInTheDocument();

    await user.click(await screen.findByRole("button", { name: "切换到演示模式" }));

    expect(screen.getByText("演示模式")).toBeInTheDocument();
    expect(drawer.queryByText("path-planner-route/v1")).not.toBeInTheDocument();
    expect(drawer.queryByText("path-feedback-summary/v1")).not.toBeInTheDocument();
    expect(drawer.queryByText(/outputs\//)).not.toBeInTheDocument();
    expect(drawer.getByText(/演示模式隐藏 schema 细节/)).toBeInTheDocument();
    expect(drawer.getByText(/禁止：full run \/ PPO \/ training/)).toBeInTheDocument();
  });

  test("presentation mode hides raw evidence after it was expanded", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => expect(screen.getAllByRole("button", { name: "路线制导" }).length).toBeGreaterThan(0));
    await user.click(firstStageButton("路线制导"));

    const panel = screen.getByRole("region", { name: "路线制导证据" });
    await user.click(within(panel).getByRole("button", { name: "查看证据链" }));

    expect(within(panel).getByRole("heading", { name: "Raw Evidence" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "切换到演示模式" }));

    expect(screen.getByText("演示模式")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Raw Evidence" })).not.toBeInTheDocument();
    expect(screen.queryByText(/stderr/)).not.toBeInTheDocument();
  });

  test("renders the mission map canvas", async () => {
    render(<App />);

    expect(await screen.findByLabelText("月面任务地图")).toBeInTheDocument();
  });

  test("mission map replay defaults to the first replay selection after route load", async () => {
    render(<App />);

    const replay = within(await screen.findByRole("region", { name: "地图回放主交互" }));
    expect(await replay.findByText(/当前对象：目标接近/)).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /t2/ })).toHaveAttribute("aria-pressed", "true");
  });

  test("mission map replay exposes layer toggles and timeline selection", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByRole("region", { name: "地图回放主交互" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "raw path" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "smoothed path" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "optimized path" })).toHaveAttribute("aria-pressed", "true");

    await user.click(screen.getByRole("button", { name: "raw path" }));
    expect(screen.getByRole("button", { name: "raw path" })).toHaveAttribute("aria-pressed", "false");

    await user.click(await screen.findByRole("button", { name: /t2/ }));
    const replay = within(screen.getByRole("region", { name: "地图回放主交互" }));
    expect(replay.getByText(/当前对象：目标接近/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /t0/ }));
    expect(replay.getByText(/当前对象：起点/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /t0/ })).toHaveAttribute("aria-pressed", "true");
  });

  test("mission map replay keeps large grid cells visible with positive finite fill rectangles", async () => {
    let invalidFillRect = false;
    let fillRectCount = 0;
    const context = {
      beginPath: vi.fn(),
      clearRect: vi.fn(),
      fill: vi.fn(),
      fillRect: vi.fn((_x: number, _y: number, width: number, height: number) => {
        fillRectCount += 1;
        if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) {
          invalidFillRect = true;
        }
      }),
      lineTo: vi.fn(),
      moveTo: vi.fn(),
      restore: vi.fn(),
      save: vi.fn(),
      stroke: vi.fn(),
      arc: vi.fn(),
      set fillStyle(_value: string) {},
      set strokeStyle(_value: string) {},
      set lineWidth(_value: number) {},
      set lineCap(_value: CanvasLineCap) {},
      set lineJoin(_value: CanvasLineJoin) {},
      set shadowColor(_value: string) {},
      set shadowBlur(_value: number) {},
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => context);

    render(
      <MissionMapReplay
        stage={routeGuidanceStage}
        sidecar={{ schema_version: "path-planner-sidecar/v1", grid: { width: 961, height: 561 }, cost: [[1]] }}
        route={null}
        frames={[{ frameId: "t0", timeLabel: "t0", label: "任务阶段", objectType: "mission-stage" }]}
        selectedFrameId="t0"
        selectedObject={{ frameId: "t0", label: "任务阶段", objectType: "mission-stage" }}
        layers={{ rawPath: true, smoothedPath: true, optimizedPath: true, blocked: true }}
        onToggleLayer={vi.fn()}
        onSelectFrame={vi.fn()}
      />,
    );

    await waitFor(() => expect(fillRectCount).toBeGreaterThan(961 * 561));
    expect(invalidFillRect).toBe(false);
  });

  test("renders mission map with malformed sidecar and route payloads without non-finite canvas coordinates", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/api/health")) {
          return Promise.resolve(new Response(JSON.stringify({ status: "ok" })));
        }
        if (url.endsWith("/api/project/status")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                repo_root: "repo",
                subprojects: {
                  "dev-platform-constraints": { exists: true },
                  "model-explorer": { exists: true },
                  "path-planner": { exists: true },
                },
              }),
            ),
          );
        }
        if (url.endsWith("/api/artifacts")) {
          return Promise.resolve(new Response(JSON.stringify({ artifacts: malformedArtifacts() })));
        }
        if (url.endsWith("/api/artifacts/bad-sidecar/raw")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                schema_version: "path-planner-sidecar/v1",
                cost: [[1, "bad", Number.NaN], "not-a-row", [Number.POSITIVE_INFINITY, 3]],
                passable_mask: [[true, false], "not-a-mask"],
                top_goals: [{ cell: ["x", 1], reachable: true }, { cell: [1, Number.NaN], reachable: false }],
              }),
            ),
          );
        }
        if (url.endsWith("/api/artifacts/bad-route/raw")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                schema_version: "path-planner-route/v1",
                geometric_path: [[0, 0], ["bad", 1], [Number.POSITIVE_INFINITY, 2], [2]],
                postprocess: { smoothed_path: [[0, 0], [1, Number.NaN], [2, 2]] },
                trajectory_optimization_report: {
                  optimized_path: [["x", "y"], [2, 2]],
                  resampled_optimized_path: [[0, 0], [Number.NEGATIVE_INFINITY, 1]],
                },
              }),
            ),
          );
        }
        return Promise.resolve(new Response("{}", { status: 404 }));
      }),
    );

    render(<App />);

    expect(await screen.findByLabelText("月面任务地图")).toBeInTheDocument();
    await waitFor(() => expect(canvasCalls.some((call) => call.method === "fillRect")).toBe(true));
    for (const call of canvasCalls) {
      expect(call.values.every(Number.isFinite)).toBe(true);
    }
  });
});

const routeGuidanceStage: DerivedMissionStage = {
  id: "route-guidance",
  label: "路线制导",
  description: "读取路径规划结果，判断是否已有可执行路线证据。",
  schemas: ["path-planner-route/v1"],
  state: "current",
  artifacts: [],
  judgment: "路线制导是当前任务叙事焦点。",
  credibility: "测试阶段对象。",
  risk: "测试阶段对象。",
  nextAction: "测试阶段对象。",
};
