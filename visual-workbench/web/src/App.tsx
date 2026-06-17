import { useEffect, useMemo, useRef, useState } from "react";

import { fetchRawJson, getJson } from "./api";
import { EvidenceChainDrawer } from "./components/EvidenceChainDrawer";
import { MissionEvidencePanel } from "./components/MissionEvidencePanel";
import { MissionKpiStrip } from "./components/MissionKpiStrip";
import { MissionMapReplay } from "./components/MissionMapReplay";
import { MissionStageRail } from "./components/MissionStageRail";
import { MissionStatusHeader } from "./components/MissionStatusHeader";
import { StageToolPanel } from "./components/StageToolPanel";
import { StageTools, type StageToolId } from "./components/StageTools";
import {
  DEFAULT_MAP_LAYERS,
  buildEvidenceChain,
  buildMissionCockpitKpis,
  buildReplayFrames,
  firstReplaySelection,
} from "./domain/missionCockpit";
import { deriveMissionStages, getCurrentStage, type DerivedMissionStage, type MissionStageId } from "./domain/missionStages";
import type { Artifact, MapLayerState, ProjectStatus, ReplayFrame, RoutePayload, SelectedMapObject, SidecarPayload } from "./types";

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
  const [mapLayers, setMapLayers] = useState<MapLayerState>(DEFAULT_MAP_LAYERS);
  const [selectedFrameId, setSelectedFrameId] = useState<string | undefined>();
  const [selectedMapObject, setSelectedMapObject] = useState<SelectedMapObject | null>(null);
  const autoSelectedStageId = useRef<MissionStageId | undefined>(undefined);
  const userSelectedStage = useRef(false);
  const userSelectedFrame = useRef(false);
  const stageResetSelection = useRef<SelectedMapObject | null>(null);

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
  const evidenceChain = useMemo(
    () => buildEvidenceChain(evidenceStage, artifacts, selectedMapObject),
    [artifacts, evidenceStage, selectedMapObject],
  );
  const cockpitKpis = useMemo(
    () => buildMissionCockpitKpis(evidenceStage, artifacts, route),
    [artifacts, evidenceStage, route],
  );
  const replayFrames = useMemo(() => buildReplayFrames(sidecar, route), [sidecar, route]);

  useEffect(() => {
    const stageResetObject = stageResetSelection.current;
    if (stageResetObject) {
      if (selectedFrameId !== stageResetObject.frameId) {
        setSelectedFrameId(stageResetObject.frameId);
      }
      if (selectedMapObject !== stageResetObject) {
        setSelectedMapObject(stageResetObject);
      }
      return;
    }

    const currentFrame = selectedFrameId ? replayFrames.find((frame) => frame.frameId === selectedFrameId) : undefined;
    const fallbackFrame = userSelectedFrame.current
      ? currentFrame ?? firstReplaySelection(replayFrames) ?? null
      : firstReplaySelection(replayFrames) ?? null;
    const nextFrameId = fallbackFrame?.frameId ?? "t0";

    if (selectedFrameId !== nextFrameId) {
      setSelectedFrameId(nextFrameId);
    }
    if (selectedMapObject !== fallbackFrame) {
      setSelectedMapObject(fallbackFrame);
    }
  }, [replayFrames, selectedFrameId, selectedMapObject]);

  function selectStage(stageId: MissionStageId) {
    const stageFallbackSelection: SelectedMapObject = { objectType: "mission-stage", frameId: "stage-reset", label: "任务阶段" };
    stageResetSelection.current = stageFallbackSelection;
    userSelectedStage.current = true;
    userSelectedFrame.current = false;
    setSelectedStageId(stageId);
    setActiveTool(null);
    setSelectedFrameId(stageFallbackSelection.frameId);
    setSelectedMapObject(stageFallbackSelection);
  }

  function toggleStageTool(toolId: StageToolId) {
    setActiveTool((currentTool) => (currentTool === toolId ? null : toolId));
  }

  function toggleMapLayer(layer: keyof MapLayerState) {
    setMapLayers((currentLayers) => ({
      ...currentLayers,
      [layer]: !currentLayers[layer],
    }));
  }

  function selectReplayFrame(frame: ReplayFrame) {
    stageResetSelection.current = null;
    userSelectedFrame.current = true;
    setSelectedFrameId(frame.frameId);
    setSelectedMapObject(frame);
  }

  return (
    <div className="app-shell mission-control-app">
      <MissionStageRail stages={missionStages} selectedStageId={selectedStage.id} onSelectStage={selectStage} />

      <main className="mission-main h1-cockpit-main">
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
            <MissionMapReplay
              stage={selectedStage}
              sidecar={sidecar}
              route={route}
              frames={replayFrames}
              selectedFrameId={selectedFrameId}
              selectedObject={selectedMapObject}
              layers={mapLayers}
              onToggleLayer={toggleMapLayer}
              onSelectFrame={selectReplayFrame}
            />
            <StageTools stage={selectedStage} activeTool={activeTool} onOpenTool={toggleStageTool} />
            {activeTool ? (
              <StageToolPanel toolId={activeTool} stage={evidenceStage} status={status} chain={evidenceChain} />
            ) : null}
          </div>
          <MissionEvidencePanel stage={evidenceStage} presentationMode={presentationMode} />
        </section>
        <EvidenceChainDrawer chain={evidenceChain} presentationMode={presentationMode} />
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

function ErrorState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="error-state" role="alert">
      <strong>{title}</strong>
      <p>{detail}</p>
    </div>
  );
}
