import type { MissionCockpitKpis } from "../types";

type MissionKpiStripProps = {
  kpis: MissionCockpitKpis;
};

export function MissionKpiStrip({ kpis }: MissionKpiStripProps) {
  const cards = [
    ["证据完整度", kpis.evidenceCompletenessLabel],
    ["关键 artifact", kpis.keyArtifactCount],
    ["回放帧", kpis.replayFrameLabel],
    ["剩余风险", kpis.riskLabel],
  ] as const;

  return (
    <section className="mission-kpi-strip" aria-label="任务态势摘要">
      {cards.map(([label, value]) => (
        <article className="mission-kpi-card" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </article>
      ))}
    </section>
  );
}
