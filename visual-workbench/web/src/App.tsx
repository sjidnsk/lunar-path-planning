import { useEffect, useMemo, useRef, useState } from "react";

import { fetchRawJson, getJson } from "./api";
import { EvidenceTraceView, ValidateView } from "./components/LegacyWorkbenchViews";
import { MissionEvidencePanel } from "./components/MissionEvidencePanel";
import { MissionKpiStrip } from "./components/MissionKpiStrip";
import { MissionMap } from "./components/MissionMap";
import { MissionStageRail } from "./components/MissionStageRail";
import { MissionStatusHeader } from "./components/MissionStatusHeader";
import { StageTools, type StageToolId } from "./components/StageTools";
import { buildMissionCockpitKpis } from "./domain/missionCockpit";
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
  const cockpitKpis = useMemo(
    () => buildMissionCockpitKpis(evidenceStage, artifacts, route),
    [artifacts, evidenceStage, route],
  );

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
        <MissionStatusHeader
          stage={selectedStage}
          health={health}
          presentationMode={presentationMode}
          onTogglePresentationMode={() => setPresentationMode((value) => !value)}
        />
        <MissionKpiStrip kpis={cockpitKpis} />

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
