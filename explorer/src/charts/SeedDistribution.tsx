import { scaleLinear } from "d3";
import type { PairedSeedDelta } from "../types/scenario";

interface SeedDistributionProps {
  rows: PairedSeedDelta[];
}

export function SeedDistribution({ rows }: SeedDistributionProps) {
  const width = 520;
  const height = 92;
  const pad = { left: 24, right: 24, top: 14, bottom: 18 };
  const values = rows.map((row) => row.profitDelta);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0);
  const x = scaleLinear().domain([min, max]).range([pad.left, width - pad.right]);
  const zero = x(0);

  return (
    <section className="chart-panel seed-panel">
      <div className="section-title">
        <span>20 Paired Seeds</span>
        <strong>{rows.filter((row) => row.priceCapProfitHigher).length}/20 positive at 40k</strong>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Paired seed profit delta distribution">
        <line className="zero" x1={zero} x2={zero} y1={pad.top} y2={height - pad.bottom} />
        {rows.map((row, index) => (
          <circle
            key={row.seed}
            cx={x(row.profitDelta)}
            cy={pad.top + 12 + (index % 4) * 13}
            r={4.5}
            className={row.priceCapProfitHigher ? "positive-dot" : "negative-dot"}
          />
        ))}
        <text x={pad.left} y={height - 3}>{min.toFixed(0)}</text>
        <text x={zero + 3} y={height - 3}>0</text>
        <text x={width - pad.right} y={height - 3} textAnchor="end">{max.toFixed(0)}</text>
      </svg>
    </section>
  );
}
