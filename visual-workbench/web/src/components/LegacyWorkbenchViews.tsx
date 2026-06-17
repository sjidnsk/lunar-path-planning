import type { Artifact } from "../types";

export function EvidenceTraceView({ artifacts }: { artifacts: Artifact[] }) {
  return (
    <div className="legacy-tool-view">
      <h3>Evidence Trace</h3>
      <p>当前阶段关联 {artifacts.length} 个 artifact；详细原始证据请在右侧渐进证据面板中展开查看。</p>
    </div>
  );
}

export function ValidateView() {
  return (
    <div className="legacy-tool-view">
      <h3>Validate</h3>
      <p>Validate remains dry-run / validate only.</p>
      <p>前端入口只允许 dry-run 或 validate，完整 run 不会从此界面触发。</p>
    </div>
  );
}
