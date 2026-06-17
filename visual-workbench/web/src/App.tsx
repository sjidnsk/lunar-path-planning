import { useEffect, useMemo, useRef, useState } from "react";
import { PanelLeft, Presentation, Signal } from "lucide-react";

import { fetchRawJson, getJson } from "./api";
import { EvidenceTraceView, ValidateView } from "./components/LegacyWorkbenchViews";
import { MissionEvidencePanel } from "./components/MissionEvidencePanel";
import { MissionMap } from "./components/MissionMap";
import {
  getStageDescription,
  getStageLabel,
  MissionStageRail,
} from "./components/MissionStageRail";
import { StageTools, type StageToolId } from "./components/StageTools";
import { deriveMissionStages, getCurrentStage, type DerivedMissionStage, type MissionStageId } from "./domain/missionStages";
import type { Artifact, ProjectStatus, RoutePayload, SidecarPayload } from "./types";

export function App() {
  const [presentationMode, setPresentationMode] = useState(false);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [status, setStatus] = useState<ProjectStatus | null>(null);
  const [health, setHealth] = useState<string>("unknown");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedStageId, setSelectedStageId] = useState<MissionStageId | undefined>();
  const [activeTool, setActiveTool] = useState<StageToolId | null>(null);
  const [sidecar, setSidecar] = useState<SidecarPayload | null>(null);
  const [route, setRoute] = useState<RoutePayload | null>(null);
  const autoSelectedStageId = useRef<MissionStageId | undefined>(undefined);
  const userSelectedStage = useRef(false);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      getJson<ProjectStatus>("/api/project/status"),
      getJson<{ artifacts: Artifact[] }>("/api/artifacts"),
      getJson<{ status: string }>("/api/health"),
    ])
      .then(([projectStatus, artifactPayload, healthPayload]) => {
        if (cancelled) return;
        setStatus(projectStatus);
        setArtifacts(artifactPayload.artifacts);
        setHealth(healthPayload.status);
        setLoadError(null);
      })
      .catch((error: Error) => {
        if (!cancelled) setLoadError(error.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const missionStages = useMemo(() => deriveMissionStages(artifacts), [artifacts]);
  const fallbackStageId = getCurrentStage(missionStages)?.id ?? missionStages[0]?.id;

  useEffect(() => {
    if (!fallbackStageId) return;
    if (!selectedStageId || !missionStages.some((stage) => stage.id === selectedStageId)) {
      setSelectedStageId(fallbackStageId);
      autoSelectedStageId.current = fallbackStageId;
      return;
    }
    if (!userSelectedStage.current && selectedStageId === autoSelectedStageId.current && selectedStageId !== fallbackStageId) {
      setSelectedStageId(fallbackStageId);
      autoSelectedStageId.current = fallbackStageId;
    }
  }, [fallbackStageId, missionStages, selectedStageId]);

  useEffect(() => {
    let cancelled = false;
    const sidecarArtifact = artifacts.find((artifact) => artifact.schema_version === "path-planner-sidecar/v1");
    const routeArtifact = artifacts.find((artifact) => artifact.schema_version === "path-planner-route/v1");

    Promise.all([
      sidecarArtifact ? fetchRawJson<SidecarPayload>(sidecarArtifact.artifact_id).catch(() => null) : Promise.resolve(null),
      routeArtifact ? fetchRawJson<RoutePayload>(routeArtifact.artifact_id).catch(() => null) : Promise.resolve(null),
    ]).then(([sidecarPayload, routePayload]) => {
      if (cancelled) return;
      setSidecar(sidecarPayload);
      setRoute(routePayload);
    });

    return () => {
      cancelled = true;
    };
  }, [artifacts]);

  const selectedStage =
    missionStages.find((stage) => stage.id === selectedStageId) ??
    missionStages.find((stage) => stage.id === fallbackStageId) ??
    missionStages[0];
  const evidenceStage = useMemo(() => enrichStageEvidence(selectedStage, artifacts), [artifacts, selectedStage]);

  function selectStage(stageId: MissionStageId) {
    userSelectedStage.current = true;
    setSelectedStageId(stageId);
    setActiveTool(null);
  }

  function toggleStageTool(toolId: StageToolId) {
    setActiveTool((currentTool) => (currentTool === toolId ? null : toolId));
  }

  return (
    <div className="app-shell mission-control-app">
      <MissionStageRail stages={missionStages} selectedStageId={selectedStage.id} onSelectStage={selectStage} />

      <main className="mission-main">
        <header className="mission-topbar">
          <span className={health === "ok" ? "connection-status ok" : "connection-status"} aria-label={`API 状态：${health}`}>
            <Signal size={14} aria-hidden="true" />
            API {health}
          </span>
          <div>
            <p className="section-label">lunar-path-planning / visual-workbench</p>
            <h1>当前月面巡视任务走到哪里了？</h1>
            <p className="mission-summary">
              {getStageLabel(selectedStage.id)}：{getStageDescription(selectedStage.id)}
            </p>
          </div>
          <div className="mission-topbar-actions">
            <span className={presentationMode ? "mode-pill presentation" : "mode-pill research"}>
              <Presentation size={16} aria-hidden="true" />
              {presentationMode ? "演示模式" : "研发模式"}
            </span>
            <button
              className="button secondary"
              type="button"
              aria-label={presentationMode ? "切换到研发模式" : "切换到演示模式"}
              onClick={() => setPresentationMode((value) => !value)}
            >
              <PanelLeft size={16} aria-hidden="true" />
              {presentationMode ? "研发视图" : "演示视图"}
            </button>
          </div>
        </header>

        {loadError ? <ErrorState title="无法连接后端 API" detail={loadError} /> : null}

        <section className="mission-shell" aria-label="任务控制台">
          <div className="mission-map-stack">
            <MissionMap stage={selectedStage} sidecar={sidecar} route={route} />
            <StageTools stage={selectedStage} activeTool={activeTool} onOpenTool={toggleStageTool} />
            {activeTool ? <StageToolPanel toolId={activeTool} stage={evidenceStage} status={status} /> : null}
          </div>
          <MissionEvidencePanel stage={evidenceStage} presentationMode={presentationMode} />
        </section>
      </main>
    </div>
  );
}

function enrichStageEvidence(stage: DerivedMissionStage, artifacts: Artifact[]): DerivedMissionStage {
  const contextSchemas =
    stage.id === "reachability-confirmation" || stage.id === "risk-review"
      ? [...stage.schemas, "path-planner-route/v1"]
      : stage.schemas;
  const contextArtifacts = artifacts.filter((artifact) => contextSchemas.includes(artifact.schema_version ?? ""));
  const deduped = Array.from(new Map([...stage.artifacts, ...contextArtifacts].map((artifact) => [artifact.artifact_id, artifact])).values());

  return {
    ...stage,
    artifacts: deduped,
  };
}

function StageToolPanel({
  toolId,
  stage,
  status,
}: {
  toolId: StageToolId;
  stage: DerivedMissionStage;
  status: ProjectStatus | null;
}) {
  return (
    <section className="stage-tool-panel" aria-label="阶段工具面板">
      {toolId === "evidence-trace" ? <EvidenceTraceView artifacts={stage.artifacts} /> : null}
      {toolId === "map-replay" ? (
        <div className="legacy-tool-view">
          <h3>Map Replay</h3>
          <p>地图回放使用当前已加载的 sidecar 与 route artifact；缺失数据时保留默认月面网格。</p>
          <p>Repo root: {status?.repo_root ?? "加载中"}</p>
        </div>
      ) : null}
      {toolId === "validate" ? <ValidateView /> : null}
    </section>
  );
}

function ErrorState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="error-state" role="alert">
      <strong>{title}</strong>
      <p>{detail}</p>
    </div>
  );
}
