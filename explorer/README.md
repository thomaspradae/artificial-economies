# Agent Economies Explorer

Static interactive replay app for thesis-facing artificial-economies results.

The browser does not train agents. It loads validated JSON bundles exported from
the Python repo and renders synchronized animation, metrics, traces, and
paired-seed evidence.

## Local Workflow

From the repository root:

```bash
python scripts/export_web_bundle.py
cd explorer
npm install
npm run build
npm run dev
```

The app has two layers:

- `all-world-ladder`: a five-world dashboard covering Pricing Arena, Resource
  Island v1, Auction House, Public Goods, and Labor Market across Q-learning,
  DQN, PPO, independent-DQN, and centralized-critic where full-run tables exist.
- `pricing-cap-reversal`: a guided Pricing Arena mechanism trace showing no
  regulation versus price cap, Q-learning versus DQN, and the DQN training-budget
  reversal.

## Data Contract

Scenario bundles live under `public/data/<scenario-id>/`.
They contain:

- real evaluation replay frames;
- benchmark metadata;
- aggregate full-run summaries;
- paired-seed deltas;
- source artifact paths.

The all-world bundle is `public/data/all-world-ladder/summary.json`. It is a
normalized summary layer over the thesis-facing `mind_comparison.csv` files.

Large raw simulator outputs should stay in the Python repo and should not be
published directly to GitHub Pages.
