import { useMemo, useState } from "react";
import { MindMetricBars } from "../charts/MindMetricBars";
import { formatNumber } from "../engine/MetricStore";
import type { CrossWorldBundle, CrossWorldMetric, CrossWorldSummary, FullLearnerId } from "../types/scenario";
import { SegmentedControl } from "./SegmentedControl";

interface CrossWorldDashboardProps {
  bundle: CrossWorldBundle;
  selectedWorldId: string;
  onWorldChange?: (worldId: string) => void;
}

const preferredMetricByWorld: Record<string, string> = {
  pricing_arena: "profit_collusion_index_mean",
  resource_island: "trade_count_mean",
  auction_house: "ex_post_regret_mean_mean",
  public_goods: "contribution_total_mean",
  labor_market: "truthful_report_rate_mean"
};

const worldPresentation: Record<
  string,
  {
    type: string;
    question: string;
    benchmark: string;
    finding: string;
    channel: string;
    limitation: string;
    accentClass: string;
  }
> = {
  auction_house: {
    type: "Auction mechanism lab",
    question: "Do learning bidders recover known auction-theory behavior?",
    benchmark: "Truthful second-price bidding, first-price shading, reserve revenue-efficiency tradeoff.",
    finding: "The mechanism structure is correct, but learned bidders remain away from benchmark bidding.",
    channel: "Regret and efficiency separate from raw bidder payoff.",
    limitation: "This tests benchmark deviation, not convergence to auction equilibrium.",
    accentClass: "accent-violet"
  },
  labor_market: {
    type: "Matching market",
    question: "Do learned worker reports preserve stability and truthfulness under deferred acceptance?",
    benchmark: "Worker-proposing deferred acceptance gives stability and proposing-side strategy-proofness.",
    finding: "Match rates stay high, while welfare, stability, and truthful reporting can separate.",
    channel: "A welfare gain can mask weaker truthfulness or stability.",
    limitation: "Manipulation claims are bounded to the worker-proposing mechanism implemented here.",
    accentClass: "accent-green"
  },
  pricing_arena: {
    type: "Regulated price game",
    question: "Can pricing institutions reduce collusion and exploitability under learning agents?",
    benchmark: "Static Nash and joint-profit prices bracket the repeated pricing game.",
    finding: "Price caps reduce exploitability broadly, but profit effects depend on architecture and training budget.",
    channel: "A guardrail can close one channel while leaving a quantity or margin channel open.",
    limitation: "The strongest claim is finite-run evidence in a stylized pricing game.",
    accentClass: "accent-orange"
  },
  public_goods: {
    type: "Commons stress test",
    question: "Do institutions prevent free-riding and resource collapse?",
    benchmark: "Free-riding and social-optimum policies bracket the common-pool problem.",
    finding: "Institutions can raise measured welfare without fully repairing contribution or sustainability.",
    channel: "Reward accounting and resource-state repair are distinct outcomes.",
    limitation: "Group-size sweeps test whether the free-riding mechanism survives scaling.",
    accentClass: "accent-yellow"
  },
  resource_island: {
    type: "Scarce trade economy",
    question: "Do property, reputation, and trade rules activate productive exchange?",
    benchmark: "No closed-form benchmark; the first obligation is institution activation under pressure.",
    finding: "Trade and property channels are interpretable only when the world creates contested access and unequal exchange.",
    channel: "Market access, spatial friction, and learner exploration jointly determine whether exchange appears.",
    limitation: "This is an activation-conditioned world, not a scalar optimum comparison.",
    accentClass: "accent-blue"
  }
};

const learnerLabels: Record<FullLearnerId, string> = {
  q_learning: "Q-learning",
  dqn: "DQN",
  ppo: "PPO",
  independent_dqn: "Independent DQN",
  centralized_critic: "Centralized critic"
};

export function CrossWorldDashboard({ bundle, selectedWorldId, onWorldChange }: CrossWorldDashboardProps) {
  const isHome = selectedWorldId === "home";
  const world = bundle.worlds.find((candidate) => candidate.id === selectedWorldId) ?? bundle.worlds[0];
  const [institutionByWorld, setInstitutionByWorld] = useState<Record<string, string>>({});
  const [metricByWorld, setMetricByWorld] = useState<Record<string, string>>({});
  const [menuCollapsed, setMenuCollapsed] = useState(false);

  const institution = institutionByWorld[world.id] ?? world.baselineInstitution;
  const metricKey = metricByWorld[world.id] ?? preferredMetricByWorld[world.id] ?? world.metrics[0]?.key;
  const metric = world.metrics.find((candidate) => candidate.key === metricKey) ?? world.metrics[0];
  const rows = useMemo(() => rowsForInstitution(world, institution), [world, institution]);
  const presentation = worldPresentation[world.id] ?? worldPresentation.pricing_arena;

  return (
    <section className="cross-world-section">
      <div className="top-stage">
        <div className={`world-nav-row ${menuCollapsed ? "collapsed" : ""}`}>
          <button
            type="button"
            className="menu-toggle"
            aria-label="Show world menu"
            onClick={() => setMenuCollapsed(false)}
          >
            <span />
            <span />
            <span />
          </button>
          <nav className="world-menu" aria-label="World selector">
            <button
              type="button"
              className={isHome ? "active" : ""}
              onClick={() => onWorldChange?.("home")}
            >
              Home
            </button>
            {bundle.worlds.map((item) => (
              <button
                key={item.id}
                type="button"
                className={!isHome && item.id === world.id ? "active" : ""}
                onClick={() => onWorldChange?.(item.id)}
              >
                {displayWorldTitle(item)}
              </button>
            ))}
            <button
              type="button"
              className="menu-close"
              aria-label="Hide world menu"
              onClick={() => setMenuCollapsed(true)}
            >
              x
            </button>
          </nav>
        </div>

        <div className="story-header compact-header">
          <div className="story-title-block">
            <h1>{isHome ? "Artificial economies" : displayWorldTitle(world)}</h1>
          </div>
          <p className="world-summary">
            {isHome
              ? "A compact interface for comparing institutions across pricing, resources, auctions, public goods, and matching."
              : world.summary}
          </p>
        </div>
      </div>

      {isHome ? (
        <div className="home-world-grid">
          {bundle.worlds.map((item) => (
            <button key={item.id} type="button" className="home-world" onClick={() => onWorldChange?.(item.id)}>
              <span>{displayWorldTitle(item)}</span>
              <strong>{item.summary}</strong>
            </button>
          ))}
        </div>
      ) : (
        <>
          <div className="controls-row cross-controls" aria-label="Cross-world controls">
        <SegmentedControl
          label={world.institutionLabel}
          value={institution}
          options={world.institutions.map((item) => ({ value: item, label: shortLabel(item) }))}
          onChange={(value) => setInstitutionByWorld((current) => ({ ...current, [world.id]: value }))}
        />
        <SegmentedControl
          label="Metric"
          value={metric.key}
          options={world.metrics.map((item) => ({ value: item.key, label: item.label }))}
          onChange={(value) => setMetricByWorld((current) => ({ ...current, [world.id]: value }))}
        />
          </div>

          <WorldBentoDashboard
            key={`${world.id}-${institution}-${metric.key}`}
            world={world}
            presentation={presentation}
            institution={institution}
            metric={metric}
            rows={rows}
          />

          <div className="cross-world-grid compact-comparison">
            <MindMetricBars rows={rows} metric={metric} />
            <MindMetricTable world={world} institution={institution} metric={metric} rows={rows} />
          </div>
        </>
      )}
    </section>
  );
}

function rowsForInstitution(world: CrossWorldSummary, institution: string) {
  const rows = world.rows.filter((row) => row.institution === institution);
  if (rows.length > 0) return rows;
  return world.rows.filter((row) => row.institution === world.baselineInstitution);
}

function WorldBentoDashboard({
  world,
  presentation,
  institution,
  metric,
  rows
}: {
  world: CrossWorldSummary;
  presentation: (typeof worldPresentation)[string];
  institution: string;
  metric: CrossWorldMetric;
  rows: ReturnType<typeof rowsForInstitution>;
}) {
  const values = rows
    .map((row) => row.metrics[metric.key])
    .filter((value) => Number.isFinite(value));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = Number.isFinite(max - min) ? max - min : 0;
  const ordered = [...rows].sort((a, b) => {
    const av = a.metrics[metric.key] ?? 0;
    const bv = b.metrics[metric.key] ?? 0;
    return metric.better === "lower" ? av - bv : bv - av;
  });
  const leader = ordered[0];
  const laggard = ordered[ordered.length - 1];

  return (
    <section className={`world-dashboard ${presentation.accentClass}`} aria-label={`${displayWorldTitle(world)} dashboard`}>
      <article className="dashboard-tile hero-tile">
        <span>{presentation.type}</span>
        <h2>{presentation.question}</h2>
        <p>{presentation.finding}</p>
      </article>

      <article className="dashboard-tile benchmark-tile">
        <span>Benchmark</span>
        <strong>{presentation.benchmark}</strong>
      </article>

      <article className="dashboard-tile metric-tile">
        <span>Selected Metric</span>
        <strong>{metric.label}</strong>
        <p>{metric.better === "context" ? "Read with mechanism context" : `${metric.better} is better`}</p>
      </article>

      <article className="dashboard-tile institution-tile">
        <span>{world.institutionLabel}</span>
        <strong>{shortLabel(institution)}</strong>
      </article>

      <article className="dashboard-tile channel-tile">
        <span>Architecture-Sensitive Channel</span>
        <strong>{presentation.channel}</strong>
      </article>

      <article className="dashboard-tile leader-tile">
        <span>Best On Metric</span>
        <strong>{leader ? learnerLabels[leader.mind] : "n/a"}</strong>
        <p>{leader ? formatMetric(leader.metrics[metric.key]) : "n/a"}</p>
      </article>

      <article className="dashboard-tile spread-tile">
        <span>Across-Mind Spread</span>
        <strong>{formatMetric(spread)}</strong>
        <p>
          {leader && laggard
            ? `${learnerLabels[leader.mind]} to ${learnerLabels[laggard.mind]}`
            : "n/a"}
        </p>
      </article>

      <article className="dashboard-tile learner-strip-tile">
        <div className="section-title dark-title">
          <span>Learner Suite</span>
          <strong>{displayWorldTitle(world)}</strong>
        </div>
        <div className="learner-dashboard-list">
          {rows.map((row) => (
            <div className="learner-dashboard-row" key={`${row.mind}-${row.institution}`}>
              <span>{learnerLabels[row.mind]}</span>
              <MetricPill value={row.metrics[metric.key]} min={min} max={max} />
            </div>
          ))}
        </div>
      </article>

      <article className="dashboard-tile caveat-tile">
        <span>Claim Boundary</span>
        <strong>{presentation.limitation}</strong>
      </article>
    </section>
  );
}

function MetricPill({ value, min, max }: { value: number | undefined; min: number; max: number }) {
  const span = max - min || 1;
  const width = Number.isFinite(value) ? Math.max(6, (((value as number) - min) / span) * 100) : 0;
  return (
    <div className="metric-pill">
      <div className="metric-pill-fill" style={{ width: `${width}%` }} />
      <strong>{formatMetric(value)}</strong>
    </div>
  );
}

function MindMetricTable({
  world,
  metric,
  rows
}: {
  world: CrossWorldSummary;
  institution: string;
  metric: CrossWorldMetric;
  rows: ReturnType<typeof rowsForInstitution>;
}) {
  return (
    <div className="chart-panel mind-table-panel">
      <div className="section-title">
        <span>Metric Sheet</span>
        <strong>{displayWorldTitle(world)}</strong>
      </div>
      <table className="mind-table">
        <thead>
          <tr>
            <th>mind</th>
            <th>{metric.label}</th>
            {world.metrics
              .filter((candidate) => candidate.key !== metric.key)
              .slice(0, 2)
              .map((candidate) => (
                <th key={candidate.key}>{candidate.label}</th>
              ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.mind}-${row.institution}`}>
              <td>{learnerLabels[row.mind]}</td>
              <td>{formatMetric(row.metrics[metric.key])}</td>
              {world.metrics
                .filter((candidate) => candidate.key !== metric.key)
                .slice(0, 2)
                .map((candidate) => (
                  <td key={candidate.key}>{formatMetric(row.metrics[candidate.key])}</td>
                ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatMetric(value: number | undefined) {
  return Number.isFinite(value) ? formatNumber(value as number) : "n/a";
}

function shortLabel(value: string) {
  return value
    .replaceAll("_", " ")
    .replace("centralized critic", "cent. critic")
    .replace("contribution matching", "matching")
    .replace("public goods ", "")
    .replace("second price", "2nd price")
    .replace("first price", "1st price");
}

function displayWorldTitle(world: CrossWorldSummary) {
  return world.title;
}
