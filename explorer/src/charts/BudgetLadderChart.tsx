import { scaleBand, scaleLinear } from "d3";
import { compactSteps, formatNumber } from "../engine/MetricStore";
import type { BudgetPoint } from "../types/scenario";

interface BudgetLadderChartProps {
  rows: BudgetPoint[];
  selectedSteps: number;
  onSelect: (steps: number) => void;
}

export function BudgetLadderChart({ rows, selectedSteps, onSelect }: BudgetLadderChartProps) {
  const width = 520;
  const height = 220;
  const pad = { top: 16, right: 18, bottom: 36, left: 58 };
  const minY = Math.min(0, ...rows.map((row) => row.profitDeltaCi95Low));
  const maxY = Math.max(0, ...rows.map((row) => row.profitDeltaCi95High));
  const x = scaleBand()
    .domain(rows.map((row) => String(row.trainingSteps)))
    .range([pad.left, width - pad.right])
    .padding(0.32);
  const y = scaleLinear().domain([minY, maxY]).nice().range([height - pad.bottom, pad.top]);
  const zeroY = y(0);

  return (
    <section className="chart-panel budget-panel">
      <div className="section-title">
        <span>Training Horizon</span>
        <strong>Cap-minus-none DQN profit</strong>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="DQN profit delta by training horizon">
        <line className="zero" x1={pad.left} x2={width - pad.right} y1={zeroY} y2={zeroY} />
        <line x1={pad.left} x2={pad.left} y1={pad.top} y2={height - pad.bottom} />
        <line x1={pad.left} x2={width - pad.right} y1={height - pad.bottom} y2={height - pad.bottom} />
        {rows.map((row) => {
          const center = (x(String(row.trainingSteps)) ?? 0) + x.bandwidth() / 2;
          const top = y(Math.max(row.profitDeltaMean, 0));
          const bottom = y(Math.min(row.profitDeltaMean, 0));
          const selected = row.trainingSteps === selectedSteps;
          return (
            <g key={row.trainingSteps} onClick={() => onSelect(row.trainingSteps)} className="budget-bar">
              <line className="ci" x1={center} x2={center} y1={y(row.profitDeltaCi95Low)} y2={y(row.profitDeltaCi95High)} />
              <rect
                className={selected ? "selected" : row.profitDeltaMean >= 0 ? "positive" : "negative"}
                x={(x(String(row.trainingSteps)) ?? 0)}
                y={top}
                width={x.bandwidth()}
                height={Math.max(2, bottom - top)}
                rx={3}
              />
              <text x={center} y={height - 12} textAnchor="middle">
                {compactSteps(row.trainingSteps)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="budget-copy">
        {rows.map((row) =>
          row.trainingSteps === selectedSteps ? (
            <span key={row.trainingSteps}>
              mean {formatNumber(row.profitDeltaMean)}, 95% CI [{formatNumber(row.profitDeltaCi95Low)},{" "}
              {formatNumber(row.profitDeltaCi95High)}], positive seeds {Math.round(row.positiveProfitShare * 100)}%
            </span>
          ) : null
        )}
      </div>
    </section>
  );
}
