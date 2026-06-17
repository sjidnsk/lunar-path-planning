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

export type MapLayerState = {
  rawPath: boolean;
  smoothedPath: boolean;
  optimizedPath: boolean;
  blocked: boolean;
};

type BaseSelectedMapObject = {
  frameId: string;
  label: string;
  artifactId?: string;
};

export type SelectedMapObject =
  | (BaseSelectedMapObject & {
      objectType: "mission-stage";
      schemaVersion?: string;
      pathIndex?: never;
      cell?: never;
    })
  | (BaseSelectedMapObject & {
      objectType: "goal";
      schemaVersion: "path-planner-sidecar/v1";
      cell: [number, number];
      pathIndex?: never;
    })
  | (BaseSelectedMapObject & {
      objectType: "path-segment";
      schemaVersion: "path-planner-route/v1";
      pathIndex: number;
      cell: [number, number];
    });

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
  selected: SelectedMapObject | ReplayFrame | null;
  supportingArtifacts: Artifact[];
  schemaFlow: Array<{ schema: string; status: "present" | "missing" }>;
  missingSchemas: string[];
  nextSafeAction: string;
  forbiddenActions: string[];
};
