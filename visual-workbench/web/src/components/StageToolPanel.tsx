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
      {toolId === "validate" ? <ValidatePanel chain={chain} /> : null}
    </section>
  );
}

function EvidenceTracePanel({ stage, chain }: { stage: DerivedMissionStage; chain: EvidenceChain }) {
  const presentSchemas = chain.schemaFlow.filter((item) => item.status === "present").length;
  const totalSchemas = chain.schemaFlow.length;
  const missingSchemas = chain.missingSchemas.length > 0 ? chain.missingSchemas.join("、") : "none";
  const forbiddenActions = chain.forbiddenActions.join(" / ");

  return (
    <div className="legacy-tool-view">
      <h3>Evidence Trace</h3>
      <p>
        schema coverage：{presentSchemas}/{totalSchemas}，当前 {stage.label} 阶段关联 {stage.artifacts.length} 个 artifact。
      </p>
      <p>抽屉负责暴露 missing schema：{missingSchemas}，并持续标记 forbidden action：{forbiddenActions}。</p>
    </div>
  );
}

function MapReplayPanel({ status, chain }: { status: ProjectStatus | null; chain: EvidenceChain }) {
  const selectedLabel = chain.selected?.label ?? "未选择对象";

  return (
    <div className="legacy-tool-view">
      <h3>Map Replay</h3>
      <p>图层与时间轴只读联动当前地图回放，选中对象：{selectedLabel}。</p>
      <p>Repo root: {status?.repo_root ?? "加载中"}</p>
    </div>
  );
}

function ValidatePanel({ chain }: { chain: EvidenceChain }) {
  const forbiddenActions = chain.forbiddenActions.join(" / ");

  return (
    <div className="legacy-tool-view">
      <h3>Validate</h3>
      <p>当前阶段工具只允许 dry-run 或 validate，不会从前端启动后端执行链。</p>
      <p>完整 run 不会从此界面触发；forbidden action：{forbiddenActions}；full run / PPO / training 均保持禁止。</p>
    </div>
  );
}
