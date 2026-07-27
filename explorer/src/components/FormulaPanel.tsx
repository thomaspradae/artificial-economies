import type { SimulationFrame } from "../types/scenario";
import { formatNumber } from "../engine/MetricStore";

interface FormulaPanelProps {
  noneFrame: SimulationFrame;
  capFrame: SimulationFrame;
}

export function FormulaPanel({ noneFrame, capFrame }: FormulaPanelProps) {
  const rows = [
    ["No regulation", noneFrame],
    ["Price cap", capFrame]
  ] as const;

  return (
    <section className="formula-panel" aria-label="Profit decomposition">
      <div className="section-title">
        <span>Mechanism Trace</span>
        <strong>Profit = (price - cost) x quantity</strong>
      </div>
      <div className="formula-rows">
        {rows.map(([label, frame]) => (
          <div className="formula-row" key={label}>
            <span>{label}</span>
            <code>
              ({formatNumber(frame.metrics.price)} - 1.00) x {formatNumber(frame.metrics.quantity)} ={" "}
              {formatNumber(frame.metrics.profit)}
            </code>
          </div>
        ))}
      </div>
    </section>
  );
}
