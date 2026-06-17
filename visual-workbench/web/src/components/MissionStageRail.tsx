import { CircleDot, Flag, MapPinned, Radar, Route, ShieldAlert } from "lucide-react";

import type { DerivedMissionStage, MissionStageId, MissionStageState } from "../domain/missionStages";

type StageSelectorProps = {
  stages: DerivedMissionStage[];
  selectedStageId: MissionStageId;
  onSelectStage: (stageId: MissionStageId) => void;
};

const STAGE_LABELS: Record<MissionStageId, string> = {
  "environment-mapping": "环境测绘",
  "target-capture": "目标捕获",
  "route-guidance": "路线制导",
  "reachability-confirmation": "可达确认",
  "risk-review": "风险复核",
  "mission-briefing": "任务简报",
};

const STAGE_DESCRIPTIONS: Record<MissionStageId, string> = {
  "environment-mapping": "地形、代价与通行边界",
  "target-capture": "候选目标与探索证据",
  "route-guidance": "路径规划与轨迹整理",
  "reachability-confirmation": "dry-run / validate 反馈",
  "risk-review": "失败、阻塞与剩余风险",
  "mission-briefing": "可汇报的任务状态",
};

const STAGE_ICONS = {
  "environment-mapping": MapPinned,
  "target-capture": Radar,
  "route-guidance": Route,
  "reachability-confirmation": CircleDot,
  "risk-review": ShieldAlert,
  "mission-briefing": Flag,
} as const;

export function getStageLabel(stageId: MissionStageId): string {
  return STAGE_LABELS[stageId];
}

export function getStageDescription(stageId: MissionStageId): string {
  return STAGE_DESCRIPTIONS[stageId];
}

export function getStageStateLabel(state: MissionStageState): string {
  if (state === "passed") return "已完成";
  if (state === "current") return "当前";
  if (state === "blocked") return "阻塞";
  return "等待";
}

export function MissionStageRail({ stages, selectedStageId, onSelectStage }: StageSelectorProps) {
  return (
    <aside className="mission-rail" aria-label="任务阶段导航">
      <div className="mission-brand">
        <span className="mission-brand-mark" aria-hidden="true">
          <Radar size={22} />
        </span>
        <div>
          <strong>Visual Workbench</strong>
          <span>Mission Control</span>
        </div>
      </div>
      <nav className="mission-stage-list" aria-label="六阶段任务">
        {stages.map((stage, index) => (
          <StageButton
            key={stage.id}
            index={index}
            stage={stage}
            selected={stage.id === selectedStageId}
            onSelectStage={onSelectStage}
          />
        ))}
      </nav>
    </aside>
  );
}

function StageButton({
  stage,
  index,
  selected,
  onSelectStage,
}: {
  stage: DerivedMissionStage;
  index: number;
  selected: boolean;
  onSelectStage: (stageId: MissionStageId) => void;
}) {
  const Icon = STAGE_ICONS[stage.id];
  const label = getStageLabel(stage.id);
  const stateLabel = getStageStateLabel(stage.state);

  return (
    <button
      type="button"
      className={`mission-stage-button ${selected ? "selected" : ""} state-${stage.state}`}
      aria-current={selected ? "step" : undefined}
      onClick={() => onSelectStage(stage.id)}
    >
      <span className="stage-index" aria-hidden="true">
        {String(index + 1).padStart(2, "0")}
      </span>
      <Icon size={18} aria-hidden="true" />
      <span className="stage-button-copy">
        <span className="stage-label">{label}</span>
        <span aria-hidden="true">{getStageDescription(stage.id)}</span>
      </span>
      <span className="stage-state" aria-hidden="true">
        {stateLabel}
      </span>
    </button>
  );
}
