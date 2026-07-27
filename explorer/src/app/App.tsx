import { useEffect, useMemo, useState } from "react";
import { BudgetLadderChart } from "../charts/BudgetLadderChart";
import { LineChart } from "../charts/LineChart";
import { SeedDistribution } from "../charts/SeedDistribution";
import { CrossWorldDashboard } from "../components/CrossWorldDashboard";
import { FormulaPanel } from "../components/FormulaPanel";
import { MetricReadout } from "../components/MetricReadout";
import { SegmentedControl } from "../components/SegmentedControl";
import { capMinusNoneFrame, compactSteps, formatNumber } from "../engine/MetricStore";
import { clampFrameIndex, nextFrameIndex } from "../engine/TimelineController";
import type { CrossWorldBundle, LearnerId, ScenarioBundle } from "../types/scenario";
import { PricingScene } from "../worlds/pricing/PricingScene";

const learnerOptions: Array<{ value: LearnerId; label: string }> = [
  { value: "dqn", label: "DQN" },
  { value: "q_learning", label: "Q-learning" }
];

const speedOptions = [
  { value: 1, label: "1x" },
  { value: 2, label: "2x" },
  { value: 4, label: "4x" }
];

interface AppProps {
  initialBundle: ScenarioBundle;
  crossWorldBundle: CrossWorldBundle;
}

export function App({ initialBundle, crossWorldBundle }: AppProps) {
  const [bundle] = useState<ScenarioBundle>(initialBundle);
  const [learner, setLearner] = useState<LearnerId>("dqn");
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(2);
  const [selectedSteps, setSelectedSteps] = useState(40000);
  const [selectedWorldId, setSelectedWorldId] = useState("home");

  const noneFrames = bundle?.traces[learner].none ?? [];
  const capFrames = bundle?.traces[learner].price_cap ?? [];
  const frameCount = Math.min(noneFrames.length, capFrames.length);
  const safeFrameIndex = clampFrameIndex(frameIndex, frameCount);
  const noneFrame = noneFrames[safeFrameIndex];
  const capFrame = capFrames[safeFrameIndex];
  const delta = noneFrame && capFrame ? capMinusNoneFrame(noneFrame, capFrame) : null;

  useEffect(() => {
    if (!playing || frameCount === 0) return;
    const interval = window.setInterval(() => {
      setFrameIndex((current) => nextFrameIndex(current, frameCount));
    }, Math.max(35, 180 / speed));
    return () => window.clearInterval(interval);
  }, [playing, speed, frameCount]);

  useEffect(() => {
    setFrameIndex((current) => clampFrameIndex(current, frameCount));
  }, [learner, frameCount]);

  const annotation = useMemo(() => {
    if (!bundle || !noneFrame) return "";
    const eligible = bundle.annotations
      .filter((item) => item.step <= noneFrame.step)
      .sort((a, b) => b.step - a.step);
    return eligible[0]?.text ?? bundle.annotations[0]?.text ?? "";
  }, [bundle, noneFrame]);

  if (!bundle || !noneFrame || !capFrame || !delta) {
    return <main className="loading">Loading validated experiment bundle...</main>;
  }

  const selectedBudget = bundle.budgetLadder.find((row) => row.trainingSteps === selectedSteps) ?? bundle.budgetLadder[0];
  const dqnSummary = bundle.summary.dqn;
  const qSummary = bundle.summary.q_learning;
  const showPricingTrace = selectedWorldId === "pricing_arena";

  return (
    <main className="app-shell">
      <CrossWorldDashboard
        bundle={crossWorldBundle}
        selectedWorldId={selectedWorldId}
        onWorldChange={setSelectedWorldId}
      />

      {showPricingTrace && (
      <>
      <section className="story-header">
        <div>
          <p className="eyebrow">Mechanism Trace</p>
          <h1>{bundle.title}</h1>
          <p>
            Replay validated Pricing Arena traces, then compare them with the paired-seed audit that resolves the
            short-run versus long-run price-cap result.
          </p>
        </div>
        <div className="source-chip">
          <span>static bundle</span>
          <strong>{bundle.experiment.evaluationFramesPerTrace} frames x 4 traces</strong>
        </div>
      </section>

      <section className="controls-row" aria-label="Experiment controls">
        <SegmentedControl label="Learner" value={learner} options={learnerOptions} onChange={setLearner} />
        <SegmentedControl label="Speed" value={speed} options={speedOptions} onChange={setSpeed} />
        <label className="range-control">
          <span>Replay frame</span>
          <input
            type="range"
            min={0}
            max={Math.max(0, frameCount - 1)}
            value={safeFrameIndex}
            onChange={(event) => setFrameIndex(Number(event.target.value))}
          />
          <strong>{safeFrameIndex + 1}/{frameCount}</strong>
        </label>
        <button className="primary-action" type="button" onClick={() => setPlaying((value) => !value)}>
          {playing ? "Pause" : "Play"}
        </button>
        <button className="secondary-action" type="button" onClick={() => setFrameIndex((value) => nextFrameIndex(value, frameCount))}>
          Step
        </button>
      </section>

      <section className="experience-grid">
        <div className="world-column">
          <PricingScene noneFrame={noneFrame} capFrame={capFrame} priceCap={bundle.benchmarks.priceCap} />
          <div className="story-note">
            <strong>What to watch:</strong> consumers flow toward firms according to realized demand. Under the cap,
            requested prices above {bundle.benchmarks.priceCap.toFixed(1)} are clipped before profit is computed.
          </div>
        </div>

        <div className="metrics-column">
          <div className="metric-grid">
            <MetricReadout label="Cap - none price" value={delta.price} tone={delta.price <= 0 ? "good" : "bad"} />
            <MetricReadout label="Cap - none quantity" value={delta.quantity} tone={delta.quantity >= 0 ? "good" : "neutral"} />
            <MetricReadout label="Cap - none profit" value={delta.profit} tone={delta.profit >= 0 ? "bad" : "good"} />
            <MetricReadout label="Cap - none welfare" value={delta.welfare} tone={delta.welfare >= 0 ? "good" : "bad"} />
          </div>
          <FormulaPanel noneFrame={noneFrame} capFrame={capFrame} />
          <LineChart label="Price Over Replay" metric="price" noneFrames={noneFrames} capFrames={capFrames} frameIndex={safeFrameIndex} />
          <LineChart label="Profit Over Replay" metric="profit" noneFrames={noneFrames} capFrames={capFrames} frameIndex={safeFrameIndex} />
        </div>
      </section>

      <section className="evidence-grid">
        <BudgetLadderChart rows={bundle.budgetLadder} selectedSteps={selectedSteps} onSelect={setSelectedSteps} />
        <div className="evidence-copy">
          <div className="section-title">
            <span>Interpretation</span>
            <strong>{compactSteps(selectedBudget.trainingSteps)} checkpoint</strong>
          </div>
          <p>
            The animated replay is a representative frozen-policy trace. The statistical claim comes from paired seed
            deltas. At {compactSteps(selectedBudget.trainingSteps)}, DQN cap-minus-none profit is{" "}
            <strong>{formatNumber(selectedBudget.profitDeltaMean)}</strong>, with{" "}
            <strong>{Math.round(selectedBudget.positiveProfitShare * 100)}%</strong> positive seeds.
          </p>
          <p>{annotation}</p>
        </div>
        <SeedDistribution rows={bundle.pairedSeedDeltas} />
      </section>

      <section className="summary-strip">
        <div>
          <span>DQN 40k none profit</span>
          <strong>{formatNumber(dqnSummary.none.profit)}</strong>
        </div>
        <div>
          <span>DQN 40k cap profit</span>
          <strong>{formatNumber(dqnSummary.price_cap.profit)}</strong>
        </div>
        <div>
          <span>Q-learning cap exploitability</span>
          <strong>{formatNumber(qSummary.price_cap.exploitability)}</strong>
        </div>
        <div>
          <span>Source artifacts</span>
          <strong>{bundle.sourceArtifacts.length}</strong>
        </div>
      </section>
      </>
      )}
    </main>
  );
}
