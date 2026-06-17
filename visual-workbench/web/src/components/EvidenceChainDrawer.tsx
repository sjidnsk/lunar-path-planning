import type { EvidenceChain } from "../types";

type EvidenceChainDrawerProps = {
  chain: EvidenceChain;
  presentationMode: boolean;
};

export function EvidenceChainDrawer({ chain, presentationMode }: EvidenceChainDrawerProps) {
  const selected = chain.selected;
  const visibleArtifacts = presentationMode ? chain.supportingArtifacts.slice(0, 3) : chain.supportingArtifacts;

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
        <ol className="schema-flow">
          {chain.schemaFlow.map((node) => (
            <li key={node.schema} className={`schema-node ${node.status === "missing" ? "missing" : ""}`}>
              <code>{node.schema}</code>
              <span>{node.status}</span>
            </li>
          ))}
        </ol>
      </section>

      <section className="evidence-chain-section">
        <p className="section-label">Artifact 支撑</p>
        {visibleArtifacts.length > 0 ? (
          <ul className="drawer-artifact-list">
            {visibleArtifacts.map((artifact) => (
              <li key={artifact.artifact_id}>
                <strong>{artifact.name}</strong>
                <span>Schema：{artifact.schema_version ?? "unknown"}</span>
                <span>Status：{artifact.status}</span>
                {!presentationMode ? <code>{artifact.relative_path}</code> : null}
              </li>
            ))}
          </ul>
        ) : (
          <p className="drawer-empty">暂无支撑 artifact</p>
        )}
        <p className="drawer-action">{chain.nextSafeAction}</p>
        <p className="drawer-guard">禁止：{chain.forbiddenActions.join(" / ")}</p>
      </section>
    </section>
  );
}
