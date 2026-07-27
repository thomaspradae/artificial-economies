import { formatNumber } from "../engine/MetricStore";

interface MetricReadoutProps {
  label: string;
  value: number;
  unit?: string;
  tone?: "neutral" | "good" | "bad";
}

export function MetricReadout({ label, value, unit = "", tone = "neutral" }: MetricReadoutProps) {
  return (
    <div className={`metric-readout ${tone}`}>
      <span>{label}</span>
      <strong>
        {formatNumber(value)}
        {unit}
      </strong>
    </div>
  );
}
