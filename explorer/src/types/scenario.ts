export type LearnerId = "q_learning" | "dqn";
export type InstitutionId = "none" | "price_cap";

export interface ScenarioBundle {
  id: string;
  version: string;
  generatedAt: string;
  title: string;
  experiment: {
    world: string;
    institutions: InstitutionId[];
    learners: LearnerId[];
    seed: number;
    nAgents: number;
    evaluationFramesPerTrace: number;
    source: string;
  };
  benchmarks: {
    nashPrice: number;
    monopolyPrice: number;
    priceCap: number;
    unitCost: number;
  };
  traces: Record<LearnerId, Record<InstitutionId, SimulationFrame[]>>;
  budgetLadder: BudgetPoint[];
  pairedSeedDeltas: PairedSeedDelta[];
  summary: Record<LearnerId, Record<InstitutionId, SummaryMetrics>>;
  annotations: StoryAnnotation[];
  sourceArtifacts: string[];
}

export interface SimulationFrame {
  step: number;
  agents: Array<{
    id: string;
    action: {
      requestedPrice: number;
      price: number;
    };
    reward: number;
    quantity: number;
    profit: number;
  }>;
  events: Array<{
    type: "price_cap_bound";
    agentId: string;
    requested: number;
    executed: number;
  }>;
  metrics: {
    price: number;
    quantity: number;
    profit: number;
    consumerSurplus: number;
    welfare: number;
    collusionIndex: number;
    priceDispersion: number;
    margin: number;
  };
}

export interface BudgetPoint {
  trainingSteps: number;
  nSeeds: number;
  profitDeltaMean: number;
  profitDeltaCi95Low: number;
  profitDeltaCi95High: number;
  quantityDeltaMean: number;
  priceDeltaMean: number;
  welfareDeltaMean: number;
  positiveProfitShare: number;
  minProfitDelta: number;
  maxProfitDelta: number;
  source: string;
}

export interface PairedSeedDelta {
  seed: number;
  profitDelta: number;
  quantityDelta: number;
  priceDelta: number;
  welfareDelta: number;
  priceCapProfitHigher: boolean;
}

export interface SummaryMetrics {
  avgPrice: number;
  profit: number;
  quantity: number;
  welfare: number;
  consumerSurplus: number;
  exploitability: number;
  collusionIndex: number;
  profitCollusionIndex: number;
}

export interface StoryAnnotation {
  id: string;
  step: number;
  text: string;
}

export type FullLearnerId = "q_learning" | "dqn" | "ppo" | "independent_dqn" | "centralized_critic";

export interface ArchitectureAxisItem {
  mind: FullLearnerId;
  label: string;
  role: string;
  assumption: string;
  whatItTests: string;
}

export interface CrossWorldMetric {
  key: string;
  label: string;
  better: "higher" | "lower" | "context";
}

export interface CrossWorldRow {
  world: string;
  mind: FullLearnerId;
  mindLabel: string;
  institution: string;
  metrics: Record<string, number>;
  sourceDir: string;
  nSeeds: number;
}

export interface CrossWorldSummary {
  id: string;
  title: string;
  summary: string;
  interpretation: string;
  sourcePath: string;
  institutionLabel: string;
  baselineInstitution: string;
  institutions: string[];
  minds: FullLearnerId[];
  metrics: CrossWorldMetric[];
  rows: CrossWorldRow[];
}

export interface CrossWorldBundle {
  id: string;
  version: string;
  generatedAt: string;
  title: string;
  subtitle: string;
  architectureAxis: ArchitectureAxisItem[];
  worlds: CrossWorldSummary[];
  coverage: {
    worldCount: number;
    mindCount: number;
    rowCount: number;
    fullRunScale: string;
  };
  sourceArtifacts: string[];
}
