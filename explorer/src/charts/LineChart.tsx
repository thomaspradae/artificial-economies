import { extent, line, scaleLinear } from "d3";
import type { SimulationFrame } from "../types/scenario";

interface LineChartProps {
  label: string;
  noneFrames: SimulationFrame[];
  capFrames: SimulationFrame[];
  frameIndex: number;
  metric: "price" | "profit";
}

export function LineChart({ label, noneFrames, capFrames, frameIndex, metric }: LineChartProps) {
  const width = 460;
  const height = 150;
  const pad = { top: 14, right: 18, bottom: 24, left: 42 };
  const noneValues = noneFrames.map((frame, index) => ({ x: index, y: frame.metrics[metric] }));
  const capValues = capFrames.map((frame, index) => ({ x: index, y: frame.metrics[metric] }));
  const allY = [...noneValues, ...capValues].map((point) => point.y);
  const [minY = 0, maxY = 1] = extent(allY);
  const ySpan = Math.max(1, maxY - minY);
  const x = scaleLinear()
    .domain([0, Math.max(1, noneFrames.length - 1)])
    .range([pad.left, width - pad.right]);
  const y = scaleLinear()
    .domain([minY - ySpan * 0.08, maxY + ySpan * 0.08])
    .range([height - pad.bottom, pad.top]);
  const path = line<{ x: number; y: number }>()
    .x((point) => x(point.x))
    .y((point) => y(point.y));
  const cursorX = x(frameIndex);

  return (
    <section className="chart-panel">
      <div className="section-title">
        <span>{label}</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label} over replay frames`}>
        <line x1={pad.left} x2={width - pad.right} y1={height - pad.bottom} y2={height - pad.bottom} />
        <line x1={pad.left} x2={pad.left} y1={pad.top} y2={height - pad.bottom} />
        <path d={path(noneValues) ?? ""} className="series none" />
        <path d={path(capValues) ?? ""} className="series cap" />
        <line className="cursor" x1={cursorX} x2={cursorX} y1={pad.top} y2={height - pad.bottom} />
        <text x={pad.left} y={height - 5}>replay</text>
        <text x={pad.left} y={14}>{maxY.toFixed(0)}</text>
        <text x={pad.left} y={height - pad.bottom - 4}>{minY.toFixed(0)}</text>
      </svg>
      <div className="legend">
        <span className="legend-item none">No regulation</span>
        <span className="legend-item cap">Price cap</span>
      </div>
    </section>
  );
}
