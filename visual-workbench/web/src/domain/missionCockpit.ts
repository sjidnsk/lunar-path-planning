import type { DerivedMissionStage } from "./missionStages";
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

const BLOCKED_STATUSES = new Set(["blocked", "failed", "error", "rejected"]);
const FEEDBACK_SCHEMAS = ["path-feedback-manifest/v1", "path-feedback-summary/v1"];
const FORBIDDEN_ACTIONS = ["full run", "PPO", "training"];

export const DEFAULT_MAP_LAYERS: MapLayerState = {
  rawPath: true,
  smoothedPath: true,
  optimizedPath: true,
  blocked: true,
};

export function buildMissionCockpitKpis(
  stage: DerivedMissionStage,
  allArtifacts: Artifact[],
  route: RoutePayload | null,
): MissionCockpitKpis {
  const requiredSchemas = uniqueStrings(stage.schemas);
  const presentSchemaCount = requiredSchemas.filter((schema) =>
    allArtifacts.some((artifact) => artifact.schema_version === schema),
  ).length;
  const completeness = requiredSchemas.length > 0 ? Math.round((presentSchemaCount / requiredSchemas.length) * 100) : 0;
  const replayFrameLabel = buildReplayFrameLabel(route);

  return {
    evidenceCompletenessLabel: `${completeness}%`,
    keyArtifactCount: stage.artifacts.length,
    replayFrameLabel,
    riskLabel: buildRiskLabel(stage.artifacts, route),
  };
}

export function buildReplayFrames(sidecar: SidecarPayload | null, route: RoutePayload | null): ReplayFrame[] {
  const routePath = normalizePath(route?.geometric_path);

  if (routePath.length > 0) {
    return routePath.map((cell, index) => ({
      objectType: "path-segment",
      frameId: `t${index}`,
      timeLabel: `t${index}`,
      label: buildPathFrameLabel(index, routePath.length),
      pathIndex: index,
      cell,
      schemaVersion: "path-planner-route/v1",
    }));
  }

  const goalCell = firstValidGoalCell(sidecar);
  if (goalCell) {
    return [
      {
        objectType: "goal",
        frameId: "t0",
        timeLabel: "t0",
        label: "候选目标",
        cell: goalCell,
        schemaVersion: "path-planner-sidecar/v1",
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

export function firstReplaySelection(frames: ReplayFrame[]): ReplayFrame | undefined {
  return frames.at(-1);
}

export function buildEvidenceChain(
  stage: DerivedMissionStage,
  allArtifacts: Artifact[],
  selected: SelectedMapObject | ReplayFrame | null,
): EvidenceChain {
  const selectedSchema = resolveSelectedSchema(selected, allArtifacts);
  const schemaFlow = uniqueStrings([...stage.schemas, selectedSchema, ...FEEDBACK_SCHEMAS]).map((schema) => ({
    schema,
    status: hasSchemaArtifact(allArtifacts, schema) ? ("present" as const) : ("missing" as const),
  }));
  const supportingArtifacts = schemaFlow.flatMap((item) =>
    allArtifacts.filter((artifact) => artifact.schema_version === item.schema),
  );
  const missingSchemas = schemaFlow.filter((item) => item.status === "missing").map((item) => item.schema);

  return {
    selected,
    supportingArtifacts,
    schemaFlow,
    missingSchemas,
    nextSafeAction: buildNextSafeAction(missingSchemas),
    forbiddenActions: [...FORBIDDEN_ACTIONS],
  };
}

function normalizePath(value: unknown): Array<[number, number]> {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((point) => {
    if (!Array.isArray(point) || point.length < 2) {
      return [];
    }
    const [x, y] = point;
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return [];
    }
    return [[x, y] as [number, number]];
  });
}

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  return values.filter((value): value is string => typeof value === "string" && value.length > 0).filter(uniqueByValue);
}

function uniqueByValue(value: string, index: number, values: string[]): boolean {
  return values.indexOf(value) === index;
}

function buildReplayFrameLabel(route: RoutePayload | null): string {
  const routePath = normalizePath(route?.geometric_path);
  return routePath.length > 0 ? `t${routePath.length - 1}` : "t0";
}

function buildRiskLabel(artifacts: Artifact[], route: RoutePayload | null): string {
  if (
    artifacts.some((artifact) => BLOCKED_STATUSES.has(artifact.status.trim().toLowerCase())) ||
    route?.reachable === false
  ) {
    return "高";
  }
  if (artifacts.length > 0) {
    return "中低";
  }
  return "证据不足";
}

function buildPathFrameLabel(index: number, pathLength: number): string {
  if (index === 0) {
    return "起点";
  }
  if (index === pathLength - 1) {
    return "目标接近";
  }
  return "路径段";
}

function firstValidGoalCell(sidecar: SidecarPayload | null): [number, number] | undefined {
  return sidecar?.top_goals?.find((goal) => normalizePath([goal.cell]).length > 0)?.cell;
}

function resolveSelectedSchema(selected: SelectedMapObject | ReplayFrame | null, allArtifacts: Artifact[]): string | undefined {
  if (!selected) {
    return undefined;
  }
  if (selected.schemaVersion) {
    return selected.schemaVersion;
  }
  if (!selected.artifactId) {
    return undefined;
  }

  const selectedArtifact = allArtifacts.find((artifact) => artifact.artifact_id === selected.artifactId);
  return selectedArtifact?.schema_version ?? undefined;
}

function hasSchemaArtifact(artifacts: Artifact[], schema: string): boolean {
  return artifacts.some((artifact) => artifact.schema_version === schema);
}

function buildNextSafeAction(missingSchemas: string[]): string {
  if (missingSchemas.includes("path-feedback-summary/v1")) {
    return "先执行 dry-run validate，补齐 path-feedback-summary/v1 后再进入任务执行。";
  }
  return "继续只读 validate 证据链，保持人工确认后再推进。";
}
