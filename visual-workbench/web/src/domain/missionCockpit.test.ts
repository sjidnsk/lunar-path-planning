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

describe("missionCockpit", () => {
  test("buildMissionCockpitKpis derives cockpit labels from selected stage evidence and route risk", () => {
    const allArtifacts = [
      artifact("path-planner-route/v1", "failed", "route.json"),
      artifact("path-planner-sidecar/v1", "passed", "sidecar.json"),
    ];
    const route: RoutePayload = {
      schema_version: "path-planner-route/v1",
      geometric_path: [
        [0, 0],
        [1, 1],
        [2, 2],
      ],
      reachable: false,
    };
    const routeStage = deriveMissionStages(allArtifacts).find((stage) => stage.id === "route-guidance");

    expect(routeStage).toBeDefined();

    const kpis = buildMissionCockpitKpis(routeStage!, allArtifacts, route);

    expect(kpis).toEqual({
      evidenceCompletenessLabel: "100%",
      keyArtifactCount: 1,
      replayFrameLabel: "t2",
      riskLabel: "高",
    });
  });

  test("buildMissionCockpitKpis keeps risk scoped to the selected stage artifacts", () => {
    const allArtifacts = [artifact("path-feedback-summary/v1", "failed", "summary.json")];
    const route: RoutePayload = {
      schema_version: "path-planner-route/v1",
      geometric_path: [],
      reachable: true,
    };
    const routeStage = deriveMissionStages(allArtifacts).find((stage) => stage.id === "route-guidance");

    expect(routeStage).toBeDefined();

    const kpis = buildMissionCockpitKpis(routeStage!, allArtifacts, route);

    expect(routeStage!.artifacts).toEqual([]);
    expect(kpis.riskLabel).toBe("证据不足");
  });

  test("buildMissionCockpitKpis uses all artifacts for completeness without changing selected risk", () => {
    const allArtifacts = [artifact("path-planner-route/v1", "reachable", "route.json")];
    const routeStage = deriveMissionStages(allArtifacts).find((stage) => stage.id === "route-guidance");

    expect(routeStage).toBeDefined();

    const kpis = buildMissionCockpitKpis({ ...routeStage!, artifacts: [] }, allArtifacts, null);

    expect(kpis.evidenceCompletenessLabel).toBe("100%");
    expect(kpis.keyArtifactCount).toBe(0);
    expect(kpis.riskLabel).toBe("证据不足");
  });

  test("buildReplayFrames derives route frames from geometric path and returns the last frame as first selection", () => {
    const sidecar: SidecarPayload = {
      schema_version: "path-planner-sidecar/v1",
      top_goals: [{ cell: [8, 8], utility: 0.92, reachable: true }],
    };
    const route: RoutePayload = {
      schema_version: "path-planner-route/v1",
      geometric_path: [
        [0, 0],
        [2, 1],
        [8, 8],
      ],
      reachable: true,
    };

    const frames = buildReplayFrames(sidecar, route);

    expect(frames).toEqual([
      {
        objectType: "path-segment",
        frameId: "t0",
        timeLabel: "t0",
        label: "起点",
        pathIndex: 0,
        cell: [0, 0],
        schemaVersion: "path-planner-route/v1",
      },
      {
        objectType: "path-segment",
        frameId: "t1",
        timeLabel: "t1",
        label: "路径段",
        pathIndex: 1,
        cell: [2, 1],
        schemaVersion: "path-planner-route/v1",
      },
      {
        objectType: "path-segment",
        frameId: "t2",
        timeLabel: "t2",
        label: "目标接近",
        pathIndex: 2,
        cell: [8, 8],
        schemaVersion: "path-planner-route/v1",
      },
    ]);
    expect(frames[0]).not.toHaveProperty("artifactId");
    expect(firstReplaySelection(frames)).toEqual(frames[2]);
  });

  test("buildReplayFrames falls back to the first sidecar goal when route has no valid geometric path", () => {
    const sidecar: SidecarPayload = {
      schema_version: "path-planner-sidecar/v1",
      top_goals: [
        { cell: [8, 8], utility: 0.92, reachable: true },
        { cell: [9, 9], utility: 0.7, reachable: true },
      ],
    };
    const route: RoutePayload = {
      schema_version: "path-planner-route/v1",
      geometric_path: [],
    };

    const frames = buildReplayFrames(sidecar, route);

    expect(frames).toEqual([
      {
        objectType: "goal",
        frameId: "t0",
        timeLabel: "t0",
        label: "候选目标",
        cell: [8, 8],
        schemaVersion: "path-planner-sidecar/v1",
      },
    ]);
    expect(frames[0]).not.toHaveProperty("artifactId");
  });

  test("buildReplayFrames falls back to a mission-stage frame when route and goals are unavailable", () => {
    const sidecar: SidecarPayload = {
      schema_version: "path-planner-sidecar/v1",
      top_goals: [],
    };

    const frames = buildReplayFrames(sidecar, null);

    expect(frames).toEqual([
      {
        objectType: "mission-stage",
        frameId: "t0",
        timeLabel: "t0",
        label: "任务阶段",
      },
    ]);
  });

  test("buildEvidenceChain reports supporting artifacts, missing feedback summary, safe action, forbidden actions, and default layers", () => {
    const allArtifacts = [
      artifact("path-planner-route/v1", "reachable", "route.json"),
      artifact("path-feedback-manifest/v1", "passed", "manifest.json"),
    ];
    const routeStage = deriveMissionStages(allArtifacts).find((stage) => stage.id === "route-guidance");
    const selected = {
      objectType: "path-segment" as const,
      frameId: "t1",
      timeLabel: "t1",
      label: "路径段",
      pathIndex: 1,
      cell: [2, 1] as [number, number],
      schemaVersion: "path-planner-route/v1" as const,
    };

    expect(routeStage).toBeDefined();

    const chain = buildEvidenceChain(routeStage!, allArtifacts, selected);

    expect(chain.selected).toEqual(selected);
    expect(chain.supportingArtifacts.map((item) => item.schema_version)).toEqual([
      "path-planner-route/v1",
      "path-feedback-manifest/v1",
    ]);
    expect(chain.schemaFlow).toEqual([
      { schema: "path-planner-route/v1", status: "present" },
      { schema: "path-feedback-manifest/v1", status: "present" },
      { schema: "path-feedback-summary/v1", status: "missing" },
    ]);
    expect(chain.missingSchemas).toEqual(["path-feedback-summary/v1"]);
    expect(chain.nextSafeAction).toContain("dry-run");
    expect(chain.nextSafeAction).toContain("validate");
    expect(chain.forbiddenActions).toEqual(["full run", "PPO", "training"]);
    expect(DEFAULT_MAP_LAYERS).toEqual({
      rawPath: true,
      smoothedPath: true,
      optimizedPath: true,
      blocked: true,
    });
  });

  test("buildEvidenceChain adds selected schemaVersion to schema flow", () => {
    const allArtifacts = [
      artifact("path-planner-route/v1", "reachable", "route.json"),
      artifact("path-feedback-manifest/v1", "passed", "manifest.json"),
    ];
    const reachabilityStage = deriveMissionStages(allArtifacts).find((stage) => stage.id === "reachability-confirmation");
    const selected = {
      objectType: "path-segment" as const,
      frameId: "t2",
      timeLabel: "t2",
      label: "目标接近",
      pathIndex: 2,
      cell: [8, 8] as [number, number],
      schemaVersion: "path-planner-route/v1" as const,
    };

    expect(reachabilityStage).toBeDefined();

    const chain = buildEvidenceChain(reachabilityStage!, allArtifacts, selected);

    expect(chain.schemaFlow).toEqual([
      { schema: "path-feedback-manifest/v1", status: "present" },
      { schema: "path-feedback-summary/v1", status: "missing" },
      { schema: "path-planner-route/v1", status: "present" },
    ]);
  });

  test("buildEvidenceChain ignores unknown artifactId instead of treating it as a schema", () => {
    const allArtifacts = [artifact("path-feedback-manifest/v1", "passed", "manifest.json")];
    const reachabilityStage = deriveMissionStages(allArtifacts).find((stage) => stage.id === "reachability-confirmation");
    const selected = {
      objectType: "mission-stage" as const,
      frameId: "t0",
      label: "任务阶段",
      artifactId: "route-1",
    };

    expect(reachabilityStage).toBeDefined();

    const chain = buildEvidenceChain(reachabilityStage!, allArtifacts, selected);

    expect(chain.schemaFlow).toEqual([
      { schema: "path-feedback-manifest/v1", status: "present" },
      { schema: "path-feedback-summary/v1", status: "missing" },
    ]);
    expect(chain.missingSchemas).not.toContain("route-1");
  });

  test("buildEvidenceChain resolves a real artifactId to its schema without leaking artifactId as schema", () => {
    const routeArtifact = {
      ...artifact("path-planner-route/v1", "reachable", "route.json"),
      artifact_id: "route-real",
    };
    const allArtifacts = [routeArtifact, artifact("path-feedback-manifest/v1", "passed", "manifest.json")];
    const reachabilityStage = deriveMissionStages(allArtifacts).find((stage) => stage.id === "reachability-confirmation");
    const selected = {
      objectType: "mission-stage" as const,
      frameId: "t0",
      label: "任务阶段",
      artifactId: "route-real",
    };

    expect(reachabilityStage).toBeDefined();

    const chain = buildEvidenceChain(reachabilityStage!, allArtifacts, selected);

    expect(chain.schemaFlow).toEqual([
      { schema: "path-feedback-manifest/v1", status: "present" },
      { schema: "path-feedback-summary/v1", status: "missing" },
      { schema: "path-planner-route/v1", status: "present" },
    ]);
    expect(chain.schemaFlow.map((item) => item.schema)).not.toContain("route-real");
    expect(chain.missingSchemas).not.toContain("route-real");
  });
});
