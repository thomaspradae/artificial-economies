import type { SimulationFrame } from "../types/scenario";

export function capMinusNoneFrame(
  noneFrame: SimulationFrame,
  capFrame: SimulationFrame
): Record<string, number> {
  return {
    price: capFrame.metrics.price - noneFrame.metrics.price,
    quantity: capFrame.metrics.quantity - noneFrame.metrics.quantity,
    profit: capFrame.metrics.profit - noneFrame.metrics.profit,
    welfare: capFrame.metrics.welfare - noneFrame.metrics.welfare,
    consumerSurplus: capFrame.metrics.consumerSurplus - noneFrame.metrics.consumerSurplus
  };
}

export function formatNumber(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "n/a";
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits
  });
}

export function compactSteps(steps: number): string {
  if (steps >= 1000) return `${steps / 1000}k`;
  return String(steps);
}
