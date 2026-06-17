import { useEffect, useRef } from "react";

import type { DerivedMissionStage } from "../domain/missionStages";
import type { RoutePayload, SidecarPayload } from "../types";
import { getStageLabel } from "./MissionStageRail";

type MissionMapProps = {
  stage: DerivedMissionStage;
  sidecar: SidecarPayload | null;
  route: RoutePayload | null;
};

const DEFAULT_COST = [
  [1, 1, 2, 3, 4, 4, 3, 2],
  [1, 2, 2, 4, 6, 5, 3, 2],
  [2, 2, 4, 7, 9, 6, 4, 2],
  [3, 4, 6, 9, 9, 7, 4, 2],
  [2, 3, 4, 6, 7, 5, 3, 1],
  [1, 2, 2, 3, 4, 3, 2, 1],
];

export function MissionMap({ stage, sidecar, route }: MissionMapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    drawMissionMap(ctx, canvas.width, canvas.height, sidecar, route);
  }, [sidecar, route]);

  return (
    <section className="mission-map-panel" aria-label="任务地图状态">
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
      <div className="mission-map-legend" aria-label="地图图例">
        <Legend color="#f97316" label="raw path" />
        <Legend color="#06b6d4" label="smoothed path" />
        <Legend color="#22c55e" label="optimized path" />
        <Legend color="#111827" label="blocked/passability" />
      </div>
    </section>
  );
}

function drawMissionMap(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  sidecar: SidecarPayload | null,
  route: RoutePayload | null,
) {
  const safeWidth = Number.isFinite(width) && width > 0 ? width : 1;
  const safeHeight = Number.isFinite(height) && height > 0 ? height : 1;
  const cost = normalizeCost(sidecar?.cost);
  const passableMask = sidecar?.passable_mask as unknown;
  const rows = Math.max(cost.length, safeDimension(sidecar?.grid?.height), matrixHeight(passableMask), 1);
  const cols = Math.max(maxRowLength(cost), safeDimension(sidecar?.grid?.width), matrixWidth(passableMask), 1);
  const cellWidth = safeWidth / cols;
  const cellHeight = safeHeight / rows;
  const maxCost = Math.max(...cost.flat().filter(Number.isFinite), 1);

  ctx.clearRect(0, 0, safeWidth, safeHeight);
  ctx.fillStyle = "#0b1017";
  ctx.fillRect(0, 0, safeWidth, safeHeight);

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const blocked = isBlocked(passableMask, row, col);
      const value = cost[row]?.[col] ?? 1;
      const intensity = Math.min(1, Math.max(0, value / maxCost));
      ctx.fillStyle = blocked ? "#111827" : blend("#d9f99d", "#334155", intensity);
      ctx.fillRect(col * cellWidth, row * cellHeight, Math.ceil(cellWidth) - 1, Math.ceil(cellHeight) - 1);
    }
  }

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

  for (const goal of normalizeGoals(sidecar?.top_goals)) {
    const [x, y] = goal.cell;
    const cx = (x + 0.5) * cellWidth;
    const cy = (y + 0.5) * cellHeight;
    ctx.fillStyle = goal.reachable === false ? "#ef4444" : "#facc15";
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(5, Math.min(cellWidth, cellHeight) * 0.18), 0, Math.PI * 2);
    ctx.fill();
  }

  drawPath(ctx, normalizePath(route?.geometric_path), cellWidth, cellHeight, "#f97316", 5);
  drawPath(ctx, normalizePath(route?.postprocess?.smoothed_path ?? route?.smoothed_path), cellWidth, cellHeight, "#06b6d4", 4);
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

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <div className="map-legend-item">
      <span style={{ background: color }} aria-hidden="true" />
      <strong>{label}</strong>
    </div>
  );
}
