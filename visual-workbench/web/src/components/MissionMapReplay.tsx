import { useEffect, useRef } from "react";

import type { DerivedMissionStage } from "../domain/missionStages";
import type { MapLayerState, ReplayFrame, RoutePayload, SelectedMapObject, SidecarPayload } from "../types";
import { getStageLabel } from "./MissionStageRail";

type MissionMapReplayProps = {
  stage: DerivedMissionStage;
  sidecar: SidecarPayload | null;
  route: RoutePayload | null;
  frames: ReplayFrame[];
  selectedFrameId?: string;
  selectedObject: SelectedMapObject | ReplayFrame | null;
  layers: MapLayerState;
  onToggleLayer: (layer: keyof MapLayerState) => void;
  onSelectFrame: (frame: ReplayFrame) => void;
};

const DEFAULT_COST = [
  [1, 1, 2, 3, 4, 4, 3, 2],
  [1, 2, 2, 4, 6, 5, 3, 2],
  [2, 2, 4, 7, 9, 6, 4, 2],
  [3, 4, 6, 9, 9, 7, 4, 2],
  [2, 3, 4, 6, 7, 5, 3, 1],
  [1, 2, 2, 3, 4, 3, 2, 1],
];

const LAYER_BUTTONS: Array<{ key: keyof MapLayerState; label: string }> = [
  { key: "rawPath", label: "raw path" },
  { key: "smoothedPath", label: "smoothed path" },
  { key: "optimizedPath", label: "optimized path" },
  { key: "blocked", label: "blocked/passability" },
];

export function MissionMapReplay({
  stage,
  sidecar,
  route,
  frames,
  selectedFrameId,
  selectedObject,
  layers,
  onToggleLayer,
  onSelectFrame,
}: MissionMapReplayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const selectedFrame = frames.find((frame) => frame.frameId === selectedFrameId);
  const selectedObjectLabel = selectedObject?.label ?? selectedFrame?.label ?? "任务阶段";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    drawMissionMap(ctx, canvas.width, canvas.height, sidecar, route, layers, selectedFrame);
  }, [layers, route, selectedFrame, sidecar]);

  return (
    <section className="mission-map-panel mission-map-replay" aria-label="地图回放主交互">
      <div className="map-panel-header">
        <div>
          <p className="section-label">Mission Map</p>
          <h2>{getStageLabel(stage.id)} / 月面巡视状态</h2>
        </div>
        <span>{route?.reachable === false ? "路线需复核" : "证据回放"}</span>
      </div>

      <div className="map-canvas-wrap">
        <canvas ref={canvasRef} width={960} height={560} aria-label="月面任务地图" />
      </div>

      <div className="map-layer-controls" aria-label="地图图层">
        {LAYER_BUTTONS.map((layer) => (
          <button
            key={layer.key}
            type="button"
            className="map-layer-button"
            aria-pressed={layers[layer.key]}
            onClick={() => onToggleLayer(layer.key)}
          >
            {layer.label}
          </button>
        ))}
      </div>

      <div className="replay-timeline" aria-label="地图回放时间线">
        {frames.map((frame) => (
          <button
            key={frame.frameId}
            type="button"
            className="replay-frame"
            aria-pressed={frame.frameId === selectedFrameId}
            onClick={() => onSelectFrame(frame)}
          >
            <span>{frame.timeLabel}</span>
            <strong>{frame.label}</strong>
          </button>
        ))}
      </div>

      <p className="selected-map-object">当前对象：{selectedObjectLabel}</p>
    </section>
  );
}

function drawMissionMap(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  sidecar: SidecarPayload | null,
  route: RoutePayload | null,
  layers: MapLayerState,
  selectedFrame?: ReplayFrame,
) {
  const safeWidth = Number.isFinite(width) && width > 0 ? width : 1;
  const safeHeight = Number.isFinite(height) && height > 0 ? height : 1;
  const cost = normalizeCost(sidecar?.cost);
  const passableMask = sidecar?.passable_mask as unknown;
  const rows = Math.max(cost.length, safeDimension(sidecar?.grid?.height), matrixHeight(passableMask), 1);
  const cols = Math.max(maxRowLength(cost), safeDimension(sidecar?.grid?.width), matrixWidth(passableMask), 1);
  const cellWidth = safeWidth / cols;
  const cellHeight = safeHeight / rows;
  const maxCost = maxMatrixValue(cost);

  ctx.clearRect(0, 0, safeWidth, safeHeight);
  ctx.fillStyle = "#0b1017";
  ctx.fillRect(0, 0, safeWidth, safeHeight);

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const blocked = layers.blocked && isBlocked(passableMask, row, col);
      const value = cost[row]?.[col] ?? 1;
      const intensity = Math.min(1, Math.max(0, value / maxCost));
      ctx.fillStyle = blocked ? "#111827" : blend("#d9f99d", "#334155", intensity);
      ctx.fillRect(col * cellWidth, row * cellHeight, visibleCellSize(cellWidth), visibleCellSize(cellHeight));
    }
  }

  drawGrid(ctx, cols, rows, cellWidth, cellHeight, safeWidth, safeHeight);
  drawGoals(ctx, normalizeGoals(sidecar?.top_goals), cellWidth, cellHeight);

  if (layers.rawPath) {
    drawPath(ctx, normalizePath(route?.geometric_path), cellWidth, cellHeight, "#f97316", 5);
  }
  if (layers.smoothedPath) {
    drawPath(ctx, normalizePath(route?.postprocess?.smoothed_path ?? route?.smoothed_path), cellWidth, cellHeight, "#06b6d4", 4);
  }
  if (layers.optimizedPath) {
    const optimizedPath = normalizePath(route?.trajectory_optimization_report?.optimized_path);
    drawPath(
      ctx,
      optimizedPath.length ? optimizedPath : normalizePath(route?.trajectory_optimization_report?.resampled_optimized_path),
      cellWidth,
      cellHeight,
      "#22c55e",
      4,
    );
  }

  drawSelectedFrame(ctx, selectedFrame, cellWidth, cellHeight);
}

function drawGrid(
  ctx: CanvasRenderingContext2D,
  cols: number,
  rows: number,
  cellWidth: number,
  cellHeight: number,
  safeWidth: number,
  safeHeight: number,
) {
  ctx.strokeStyle = "rgba(226, 232, 240, 0.16)";
  ctx.lineWidth = 1;
  for (let col = 0; col <= cols; col += 1) {
    ctx.beginPath();
    ctx.moveTo(col * cellWidth, 0);
    ctx.lineTo(col * cellWidth, safeHeight);
    ctx.stroke();
  }
  for (let row = 0; row <= rows; row += 1) {
    ctx.beginPath();
    ctx.moveTo(0, row * cellHeight);
    ctx.lineTo(safeWidth, row * cellHeight);
    ctx.stroke();
  }
}

function drawGoals(
  ctx: CanvasRenderingContext2D,
  goals: Array<{ cell: [number, number]; reachable?: boolean }>,
  cellWidth: number,
  cellHeight: number,
) {
  for (const goal of goals) {
    const [x, y] = goal.cell;
    const cx = (x + 0.5) * cellWidth;
    const cy = (y + 0.5) * cellHeight;
    if (!Number.isFinite(cx) || !Number.isFinite(cy)) continue;
    ctx.fillStyle = goal.reachable === false ? "#ef4444" : "#facc15";
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(5, Math.min(cellWidth, cellHeight) * 0.18), 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawSelectedFrame(
  ctx: CanvasRenderingContext2D,
  selectedFrame: ReplayFrame | undefined,
  cellWidth: number,
  cellHeight: number,
) {
  if (!selectedFrame?.cell) return;
  const [x, y] = selectedFrame.cell;
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;
  const cx = (x + 0.5) * cellWidth;
  const cy = (y + 0.5) * cellHeight;
  if (!Number.isFinite(cx) || !Number.isFinite(cy)) return;

  ctx.save();
  ctx.strokeStyle = "#f8fafc";
  ctx.lineWidth = 3;
  ctx.shadowColor = "#f8fafc";
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.arc(cx, cy, Math.max(7, Math.min(cellWidth, cellHeight) * 0.28), 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function normalizeCost(value: unknown): number[][] {
  if (!Array.isArray(value)) return DEFAULT_COST;
  const rows = value
    .filter((row): row is unknown[] => Array.isArray(row))
    .map((row) => row.map((cell) => (typeof cell === "number" && Number.isFinite(cell) ? cell : 1)));
  return rows.length && maxRowLength(rows) > 0 ? rows : DEFAULT_COST;
}

function normalizePath(value: unknown): Array<[number, number]> {
  if (!Array.isArray(value)) return [];
  return value.flatMap((point) => {
    if (!Array.isArray(point) || point.length < 2) return [];
    const [x, y] = point;
    return typeof x === "number" && Number.isFinite(x) && typeof y === "number" && Number.isFinite(y) ? [[x, y] as [number, number]] : [];
  });
}

function normalizeGoals(value: unknown): Array<{ cell: [number, number]; reachable?: boolean }> {
  if (!Array.isArray(value)) return [];
  return value.flatMap((goal) => {
    if (!goal || typeof goal !== "object") return [];
    const candidate = goal as { cell?: unknown; reachable?: unknown };
    const path = normalizePath([candidate.cell]);
    if (!path.length) return [];
    return [{ cell: path[0], reachable: typeof candidate.reachable === "boolean" ? candidate.reachable : undefined }];
  });
}

function safeDimension(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? Math.floor(value) : 0;
}

function matrixHeight(value: unknown): number {
  return Array.isArray(value) ? value.filter(Array.isArray).length : 0;
}

function matrixWidth(value: unknown): number {
  if (!Array.isArray(value)) return 0;
  return maxRowLength(value.filter((row): row is unknown[] => Array.isArray(row)));
}

function maxRowLength(rows: unknown[][]): number {
  return rows.reduce((max, row) => Math.max(max, row.length), 0);
}

function maxMatrixValue(rows: number[][]): number {
  let max = 1;
  for (const row of rows) {
    for (const value of row) {
      if (Number.isFinite(value) && value > max) {
        max = value;
      }
    }
  }
  return max;
}

function visibleCellSize(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return 1;
  }
  return value > 2 ? value - 1 : value;
}

function isBlocked(mask: unknown, row: number, col: number): boolean {
  if (!Array.isArray(mask)) return false;
  const maskRow = mask[row];
  return Array.isArray(maskRow) ? maskRow[col] === false : false;
}

function drawPath(
  ctx: CanvasRenderingContext2D,
  path: Array<[number, number]>,
  cellWidth: number,
  cellHeight: number,
  color: string,
  lineWidth: number,
) {
  if (path.length === 0) return;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;
  ctx.beginPath();
  path.forEach(([x, y], index) => {
    const px = (x + 0.5) * cellWidth;
    const py = (y + 0.5) * cellHeight;
    if (!Number.isFinite(px) || !Number.isFinite(py)) return;
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.restore();
}

function blend(start: string, end: string, amount: number) {
  const a = hexToRgb(start);
  const b = hexToRgb(end);
  const r = Math.round(a[0] + (b[0] - a[0]) * amount);
  const g = Math.round(a[1] + (b[1] - a[1]) * amount);
  const blue = Math.round(a[2] + (b[2] - a[2]) * amount);
  return `rgb(${r}, ${g}, ${blue})`;
}

function hexToRgb(hex: string): [number, number, number] {
  return [1, 3, 5].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16)) as [number, number, number];
}
