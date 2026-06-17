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
  const modeLabel = presentationMode ? "演示模式" : "研发模式";
  const toggleLabel = presentationMode ? "切换到研发模式" : "切换到演示模式";

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

      <div className="mission-status-actions">
        <span className={presentationMode ? "mode-pill presentation" : "mode-pill research"}>
          <Presentation size={16} aria-hidden="true" />
          {modeLabel}
        </span>
        <button className="button secondary" type="button" aria-label={toggleLabel} onClick={onTogglePresentationMode}>
          <PanelLeft size={16} aria-hidden="true" />
          {presentationMode ? "研发视图" : "演示视图"}
        </button>
      </div>
    </header>
  );
}
