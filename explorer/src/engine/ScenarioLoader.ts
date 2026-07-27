import type { CrossWorldBundle, ScenarioBundle } from "../types/scenario";

const SCENARIO_URL = `${import.meta.env.BASE_URL}data/pricing-cap-reversal/scenario.json`;
const CROSS_WORLD_URL = `${import.meta.env.BASE_URL}data/all-world-ladder/summary.json`;

export async function loadPricingCapScenario(): Promise<ScenarioBundle> {
  const response = await fetch(SCENARIO_URL);
  if (!response.ok) {
    throw new Error(`Failed to load scenario bundle: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as ScenarioBundle;
}

export async function loadCrossWorldBundle(): Promise<CrossWorldBundle> {
  const response = await fetch(CROSS_WORLD_URL);
  if (!response.ok) {
    throw new Error(`Failed to load cross-world bundle: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as CrossWorldBundle;
}
