import { ChevronDown, ChevronRight, ShieldCheck } from "lucide-react";
import { useState } from "react";

import type { DerivedMissionStage } from "../domain/missionStages";
import { getStageDescription, getStageLabel, getStageStateLabel } from "./MissionStageRail";

type MissionEvidencePanelProps = {
  stage: DerivedMissionStage;
  presentationMode: boolean;
};

export function MissionEvidencePanel({ stage, presentationMode }: MissionEvidencePanelProps) {
  const [expanded, setExpanded] = useState(false);
  const stageLabel = getStageLabel(stage.id);
  const stateLabel = getStageStateLabel(stage.state);
  const artifacts = presentationMode ? stage.artifacts.slice(0, 3) : stage.artifacts;

  return (
    <section className="mission-evidence-panel" aria-label={`${stageLabel}证据`}>
      <div className="evidence-kicker">
        <span>{stageLabel}</span>
        <strong>{stateLabel}</strong>
      </div>

      <div className="evidence-primary">
        <p className="section-label">阶段判断</p>
        <h2>{buildJudgment(stage)}</h2>
        <p>{getStageDescription(stage.id)}</p>
      </div>

      <dl className="evidence-facts">
        <div>
          <dt>可信度</dt>
          <dd>{buildCredibility(stage)}</dd>
        </div>
        <div>
          <dt>风险/未完成边界</dt>
          <dd>{buildRisk(stage)}</dd>
        </div>
        <div>
          <dt>下一步动作</dt>
          <dd>{buildNextAction(stage)}</dd>
        </div>
      </dl>

      {!presentationMode ? (
        <button className="evidence-expand" type="button" aria-expanded={expanded} onClick={() => setExpanded((value) => !value)}>
          {expanded ? <ChevronDown size={16} aria-hidden="true" /> : <ChevronRight size={16} aria-hidden="true" />}
          {expanded ? "隐藏证据链" : "查看证据链"}
        </button>
      ) : (
        <div className="presentation-note">
          <ShieldCheck size={16} aria-hidden="true" />
          演示模式仅保留阶段判断与关键可信度。
        </div>
      )}

      {expanded && !presentationMode ? (
        <div className="raw-evidence">
          <h3>Raw Evidence</h3>
          {artifacts.length ? (
            <ul>
              {artifacts.map((artifact) => (
                <li key={artifact.artifact_id}>
                  <strong>{artifact.name}</strong>
                  <span>{artifact.schema_version ?? artifact.kind}</span>
                  <span>{artifact.status}</span>
                  {!presentationMode ? <code>{artifact.relative_path}</code> : null}
                </li>
              ))}
            </ul>
          ) : (
            <p>当前阶段还没有关联 artifact。</p>
          )}
        </div>
      ) : null}
    </section>
  );
}

function buildJudgment(stage: DerivedMissionStage): string {
  const label = getStageLabel(stage.id);
  if (stage.state === "blocked") return `${label}存在阻塞证据，任务推进需要先复核该阶段输出。`;
  if (stage.state === "current") return `${label}是当前任务焦点，需要用现有证据确认能否进入下一阶段。`;
  if (stage.state === "passed") return `${label}已有可追踪证据支撑，可作为后续判断的输入。`;
  return `${label}尚未形成足够证据，当前只能保持等待状态。`;
}

function buildCredibility(stage: DerivedMissionStage): string {
  if (stage.artifacts.length === 0) return `尚未关联 ${stage.schemas.join(" / ")}。`;
  const schemas = Array.from(new Set(stage.artifacts.map((artifact) => artifact.schema_version ?? artifact.kind))).join(" / ");
  return `已关联 ${stage.artifacts.length} 个 artifact，覆盖 ${schemas}。`;
}

function buildRisk(stage: DerivedMissionStage): string {
  const blocked = stage.artifacts.filter((artifact) => ["blocked", "failed", "error", "rejected"].includes(artifact.status.toLowerCase()));
  if (blocked.length > 0) return `发现 ${blocked.length} 个阻塞或失败状态，不能把该阶段当作训练或放行依据。`;
  if (stage.artifacts.length === 0) return "证据缺口仍在，阶段判断不能外推为完整任务完成。";
  return "未发现阻塞状态，但仍只代表 artifact 层面的阶段证据。";
}

function buildNextAction(stage: DerivedMissionStage): string {
  if (stage.state === "blocked") return "先定位失败 artifact，再重新生成或导入对应证据。";
  if (stage.id === "reachability-confirmation") return "只通过 dry-run 或 validate 复核可达性边界。";
  if (stage.id === "risk-review") return "汇总失败、回退与未完成项，避免扩大结论。";
  if (stage.state === "pending") return `补齐 ${stage.schemas.join(" / ")} 后再判断。`;
  return "继续检查下一阶段所需证据，不触发完整 run。";
}
