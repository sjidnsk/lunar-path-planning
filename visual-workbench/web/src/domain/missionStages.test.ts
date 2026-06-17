import { describe, expect, test } from "vitest";

import type { Artifact } from "../types";
import { deriveMissionStages, getCurrentStage, MISSION_STAGES } from "./missionStages";

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

describe("missionStages", () => {
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

  test("selects environment mapping as current when no artifacts exist", () => {
    const stages = deriveMissionStages([]);

    expect(stages.find((stage) => stage.id === "environment-mapping")?.state).toBe("current");
    expect(getCurrentStage(stages)?.id).toBe("environment-mapping");
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
    expect(getCurrentStage(stages)?.id).toBe("route-guidance");
  });

  test("marks feedback validation as blocked when summary is blocked", () => {
    const stages = deriveMissionStages([
      artifact("path-planner-route/v1", "reachable", "route.json"),
      artifact("path-feedback-summary/v1", "blocked", "summary.json"),
    ]);
    const reachabilityStage = stages.find((stage) => stage.id === "reachability-confirmation");

    expect(reachabilityStage?.state).toBe("blocked");
    expect(reachabilityStage?.risk).toContain("blocked");
  });

  test("does not invent a current stage when the first stage is blocked", () => {
    const stages = deriveMissionStages([artifact("model-explorer-contract/v1", "blocked")]);

    expect(stages.find((stage) => stage.id === "environment-mapping")?.state).toBe("blocked");
    expect(stages.some((stage) => stage.state === "current")).toBe(false);
    expect(getCurrentStage(stages)).toBeUndefined();
  });

  test("selects mission briefing as current when all stages have evidence", () => {
    const stages = deriveMissionStages([
      artifact("model-explorer-contract/v1"),
      artifact("path-planner-sidecar/v1"),
      artifact("model-explorer-experiment/v1"),
      artifact("path-planner-route/v1", "reachable", "route.json"),
      artifact("path-feedback-manifest/v1"),
      artifact("path-feedback-summary/v1", "ready", "summary.json"),
    ]);

    expect(stages.find((stage) => stage.id === "mission-briefing")?.state).toBe("current");
    expect(getCurrentStage(stages)?.id).toBe("mission-briefing");
  });

  test("returns undefined when current stage is requested from an empty list", () => {
    expect(getCurrentStage([])).toBeUndefined();
  });
});
