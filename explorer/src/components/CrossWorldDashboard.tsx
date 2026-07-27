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
  resource_island_v1: "trade_count_mean",
  auction_house: "ex_post_regret_mean_mean",
  public_goods: "contribution_total_mean",
  labor_market: "truthful_report_rate_mean"
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

          <div className="cross-world-grid">
        <article className="world-brief">
          <div className="section-title">
            <span>{displayWorldTitle(world)}</span>
            <strong>{institution}</strong>
          </div>
          <p>{world.interpretation}</p>
        </article>
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
  return world.id === "resource_island_v1" ? "Resource Island" : world.title;
}
