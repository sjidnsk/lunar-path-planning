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
    description: "确认场景契约与代价地图证据已经形成。",
    schemas: ["model-explorer-contract/v1", "path-planner-sidecar/v1"],
  },
  {
    id: "target-capture",
    label: "目标捕获",
    description: "汇总探索契约与实验记录，确认候选目标来源。",
    schemas: ["model-explorer-contract/v1", "model-explorer-experiment/v1"],
  },
  {
    id: "route-guidance",
    label: "路线制导",
    description: "读取路径规划结果，判断是否已有可执行路线证据。",
    schemas: ["path-planner-route/v1"],
  },
  {
    id: "reachability-confirmation",
    label: "可达确认",
    description: "使用反馈清单与汇总结果确认路线可达性。",
    schemas: ["path-feedback-manifest/v1", "path-feedback-summary/v1"],
  },
  {
    id: "risk-review",
    label: "风险复核",
    description: "复核反馈汇总中的阻塞、失败与剩余风险。",
    schemas: ["path-feedback-summary/v1"],
  },
  {
    id: "mission-briefing",
    label: "任务简报",
    description: "合并探索实验与反馈汇总，形成任务简报依据。",
    schemas: ["model-explorer-experiment/v1", "path-feedback-summary/v1"],
  },
];

const BLOCKED_STATUSES = new Set(["blocked", "failed", "error", "rejected"]);
const PASSED_STATUSES = new Set(["passed", "ready", "reachable"]);

function normalizedStatus(artifact: Artifact): string {
  return artifact.status.trim().toLowerCase();
}

function deriveBaseState(artifacts: Artifact[]): Exclude<MissionStageState, "current"> {
  if (artifacts.some((item) => BLOCKED_STATUSES.has(normalizedStatus(item)))) {
    return "blocked";
  }
  if (artifacts.some((item) => PASSED_STATUSES.has(normalizedStatus(item))) || artifacts.length > 0) {
    return "passed";
  }
  return "pending";
}

function formatSchemas(stage: MissionStageDefinition): string {
  return stage.schemas.join("、");
}

function formatStatuses(artifacts: Artifact[]): string {
  return artifacts.map((item) => `${item.schema_version ?? "unknown"}=${item.status}`).join("、");
}

function buildNarrative(
  stage: MissionStageDefinition,
  artifacts: Artifact[],
  state: Exclude<MissionStageState, "current">,
): Pick<DerivedMissionStage, "judgment" | "credibility" | "risk" | "nextAction"> {
  const schemas = formatSchemas(stage);
  const artifactCount = artifacts.length;
  const statuses = formatStatuses(artifacts);

  if (state === "blocked") {
    return {
      judgment: `${stage.label}发现阻塞证据，需先处理反馈状态。`,
      credibility: `已关联${artifactCount}个 artifact：${statuses}。`,
      risk: `阻塞状态来自 artifact status：${statuses}。`,
      nextAction: `修复${stage.label}相关输出后重新生成 ${schemas}。`,
    };
  }

  if (state === "passed") {
    return {
      judgment: `${stage.label}已有证据支撑，可进入后续叙事判断。`,
      credibility: `已关联${artifactCount}个 artifact，覆盖状态：${statuses}。`,
      risk: "未发现阻塞状态，仍需以后续阶段证据交叉确认。",
      nextAction: `继续检查下一阶段所需 schema：${schemas}。`,
    };
  }

  return {
    judgment: `${stage.label}尚未形成可用证据。`,
    credibility: `当前未关联 artifact，期待 schema：${schemas}。`,
    risk: "证据缺失，阶段判断只能保持待确认。",
    nextAction: `生成或导入 ${schemas} 对应 artifact。`,
  };
}

export function deriveMissionStages(artifacts: Artifact[]): DerivedMissionStage[] {
  const derivedStages = MISSION_STAGES.map((stage) => {
    const groupedArtifacts = artifacts.filter((artifact) => stage.schemas.includes(artifact.schema_version ?? ""));
    const state = deriveBaseState(groupedArtifacts);

    return {
      ...stage,
      state,
      artifacts: groupedArtifacts,
      ...buildNarrative(stage, groupedArtifacts, state),
    };
  });

  const firstIncompleteIndex = derivedStages.findIndex(
    (stage) => stage.state === "pending" || stage.state === "blocked",
  );
  const currentIndex = firstIncompleteIndex === -1 ? derivedStages.length - 1 : Math.max(0, firstIncompleteIndex - 1);

  return derivedStages.map((stage, index) => {
    if (index === currentIndex && stage.state !== "blocked") {
      return {
        ...stage,
        state: "current",
        judgment: `${stage.label}是当前任务叙事焦点。${stage.judgment}`,
        nextAction: `${stage.nextAction} 当前应优先完成本阶段确认。`,
      };
    }
    return stage;
  });
}

export function getCurrentStage(stages: DerivedMissionStage[]): DerivedMissionStage | undefined {
  return stages.find((stage) => stage.state === "current");
}
