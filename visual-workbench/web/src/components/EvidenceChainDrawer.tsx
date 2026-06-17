import type { EvidenceChain } from "../types";

type EvidenceChainDrawerProps = {
  chain: EvidenceChain;
  presentationMode: boolean;
};

export function EvidenceChainDrawer({ chain, presentationMode }: EvidenceChainDrawerProps) {
  const selected = chain.selected;
  const visibleArtifacts = presentationMode ? chain.supportingArtifacts.slice(0, 3) : chain.supportingArtifacts;
  const presentNodeCount = chain.schemaFlow.filter((node) => node.status === "present").length;
  const missingNodeCount = chain.schemaFlow.filter((node) => node.status === "missing").length;
  const actionText = presentationMode ? "先补齐缺失证据并通过 dry-run validate 后再推进。" : chain.nextSafeAction;

  return (
    <section className="evidence-chain-drawer" aria-label="证据链抽屉">
      <section className="evidence-chain-section">
        <p className="section-label">Selected Map Object</p>
        <h2>当前对象：{selected?.label ?? "未选择"}</h2>
        <dl className="drawer-object-facts">
          <div>
            <dt>Object Type</dt>
            <dd>{selected?.objectType ?? "unknown"}</dd>
          </div>
          <div>
            <dt>Frame ID</dt>
            <dd>{selected?.frameId ?? "none"}</dd>
          </div>
          {!presentationMode && selected?.schemaVersion ? (
            <div>
              <dt>Schema Version</dt>
              <dd>schemaVersion：{selected.schemaVersion}</dd>
            </div>
          ) : null}
          {!presentationMode && selected?.artifactId ? (
            <div>
              <dt>Artifact ID</dt>
              <dd>artifactId：{selected.artifactId}</dd>
            </div>
          ) : null}
        </dl>
      </section>

      <section className="evidence-chain-section">
        <p className="section-label">Schema Flow</p>
        {presentationMode ? (
          <div className="drawer-summary-card">
            <strong>已连接 {presentNodeCount} 个证据节点</strong>
            <span>缺失 {missingNodeCount} 项</span>
            <p>演示模式隐藏 schema 细节</p>
          </div>
        ) : (
          <ol className="schema-flow">
            {chain.schemaFlow.map((node) => (
              <li key={node.schema} className={`schema-node ${node.status === "missing" ? "missing" : ""}`}>
                <code>{node.schema}</code>
                <span>{node.status}</span>
              </li>
            ))}
          </ol>
        )}
      </section>

      <section className="evidence-chain-section">
        <p className="section-label">Artifact 支撑</p>
        {visibleArtifacts.length > 0 ? (
          <ul className="drawer-artifact-list">
            {visibleArtifacts.map((artifact) => (
              <li key={artifact.artifact_id}>
                <strong>{artifact.name}</strong>
                {!presentationMode ? <span>Schema：{artifact.schema_version ?? "unknown"}</span> : null}
                <span>Status：{artifact.status}</span>
                {!presentationMode ? <code>{artifact.relative_path}</code> : null}
              </li>
            ))}
          </ul>
        ) : (
          <p className="drawer-empty">暂无支撑 artifact</p>
        )}
        <p className="drawer-action">{actionText}</p>
        <p className="drawer-guard">禁止：{chain.forbiddenActions.join(" / ")}</p>
      </section>
    </section>
  );
}
