import type { CrossWorldMetric, CrossWorldRow } from "../types/scenario";
import { formatNumber } from "../engine/MetricStore";

interface MindMetricBarsProps {
  rows: CrossWorldRow[];
  metric: CrossWorldMetric;
}

const mindClassNames: Record<string, string> = {
  q_learning: "mind-q",
  dqn: "mind-dqn",
  ppo: "mind-ppo",
  independent_dqn: "mind-independent",
  centralized_critic: "mind-centralized"
};

export function MindMetricBars({ rows, metric }: MindMetricBarsProps) {
  const values = rows
    .map((row) => row.metrics[metric.key])
    .filter((value) => Number.isFinite(value));
  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const span = max - min || 1;

  return (
    <div className="chart-panel mind-bars-panel">
      <div className="section-title">
        <span>All-Mind Comparison</span>
        <strong>{metric.label}</strong>
      </div>
      <div className="mind-bars">
        {rows.map((row) => {
          const value = row.metrics[metric.key];
          const width = Number.isFinite(value) ? Math.max(2, ((value - min) / span) * 100) : 0;
          return (
            <div className="mind-bar-row" key={`${row.mind}-${row.institution}`}>
              <span>{row.mindLabel}</span>
              <div className="mind-bar-track">
                <div className={`mind-bar ${mindClassNames[row.mind] ?? ""}`} style={{ width: `${width}%` }} />
              </div>
              <strong>{Number.isFinite(value) ? formatNumber(value) : "n/a"}</strong>
            </div>
          );
        })}
      </div>
      <p className="dashboard-note">
        Direction: {metric.better === "context" ? "interpret with mechanism context" : `${metric.better} is better`}.
      </p>
    </div>
  );
}
