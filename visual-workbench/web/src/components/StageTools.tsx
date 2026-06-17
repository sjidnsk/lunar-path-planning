import { FileSearch, Map, ShieldCheck } from "lucide-react";

import type { DerivedMissionStage } from "../domain/missionStages";
import { getStageLabel } from "./MissionStageRail";

export type StageToolId = "evidence-trace" | "map-replay" | "validate";

type StageToolsProps = {
  stage: DerivedMissionStage;
  activeTool: StageToolId | null;
  onOpenTool: (toolId: StageToolId) => void;
};

const stageTools: Array<{
  id: StageToolId;
  label: string;
  Icon: typeof FileSearch;
}> = [
  { id: "evidence-trace", label: "Evidence Trace", Icon: FileSearch },
  { id: "map-replay", label: "Map Replay", Icon: Map },
  { id: "validate", label: "Validate", Icon: ShieldCheck },
];

export function StageTools({ stage, activeTool, onOpenTool }: StageToolsProps) {
  return (
    <section className="stage-tools" aria-label={`${getStageLabel(stage.id)}阶段工具`}>
      {stageTools.map(({ id, label, Icon }) => {
        const isActive = activeTool === id;
        return (
          <button
            key={id}
            type="button"
            className={isActive ? "stage-tool-button active" : "stage-tool-button"}
            aria-expanded={isActive}
            aria-pressed={isActive}
            onClick={() => onOpenTool(id)}
          >
            <Icon size={18} aria-hidden="true" />
            <span>{label}</span>
          </button>
        );
      })}
    </section>
  );
}
