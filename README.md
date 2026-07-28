# Artificial Economies

Artificial Economies is a research codebase for testing whether economic institutions remain robust when their participants are learning agents rather than equilibrium solvers.

The repository accompanies the paper:

- `paper/main.tex`
- `paper/main.pdf`

Project repository:

- <https://github.com/thomaspradae/artificial-economies>

## Overview

The codebase implements five controlled economic worlds behind a shared interface:

- `Auction House`: single-item auctions with first-price, second-price, reserve, clock, and information variants.
- `Labor Market`: worker-proposing deferred-acceptance matching with learned worker reports.
- `Pricing Arena`: repeated pricing games with regulatory institutions and exploitability checks.
- `Public Goods`: common-pool contribution and extraction with penalties, matching, reputation, information restriction, and taxes.
- `Resource Island`: spatial gather-and-trade economy with property rights, trade controls, reputation, and activation diagnostics.

Each world can be paired with multiple learner classes:

- random behavior
- tabular Q-learning
- DQN
- PPO
- decorrelated independent-DQN
- centralized-critic training

The main research question is not which learner performs best. The question is which institutional guarantees survive when the behavioral assumptions behind classical benchmarks are changed.

## Repository Layout

```text
core/                 Shared World, Agent, Institution, metrics, registry, logging
institutions/         Institution implementations used across worlds
minds/                Random, tabular, deep-RL, and MARL learner implementations
worlds/               Auction House, Labor Market, Pricing Arena, Public Goods, Resource Island
paper/                Paper source, bibliography, and compiled PDF
explorer/             Interactive browser explainer for selected results
run_*.py              Experiment, validation, audit, and comparison runners
build_*.py            Table and synthesis builders
test_*.py             Unit, integration, and output-schema tests
```

Generated run outputs are intentionally not tracked in git. Experiment scripts write CSV, JSON, Markdown, and plot outputs under `outputs/`.

## Installation

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is used for the DQN, PPO, independent-DQN, and centralized-critic learners. CPU execution is sufficient for the structured MLP experiments in this repository.

## Tests

Run the full Python test suite:

```bash
python -m unittest discover
```

The tests cover:

- core abstractions
- world mechanics
- benchmark and known-answer checks
- institution transforms
- tabular and deep-RL learner integration
- output schemas and comparison builders

## Common Experiment Runners

Pricing Arena:

```bash
python run_multiseed.py --mind q_learning --steps 40000 --n-seeds 20 --save-dir outputs/pricing_q_learning
python run_exploitability.py --incumbent-mind q_learning --save-dir outputs/pricing_exploitability
```

Auction House:

```bash
python run_auction_house_smoke.py --steps 40000 --n-seeds 20 --save-dir outputs/auction_house_full
```

Public Goods:

```bash
python run_public_goods_smoke.py --steps 40000 --n-seeds 20 --save-dir outputs/public_goods_full
```

Labor Market:

```bash
python run_labor_market_smoke.py --steps 40000 --n-seeds 20 --save-dir outputs/labor_market_full
```

Resource Island:

```bash
python run_resource_island_smoke.py --steps 40000 --n-seeds 20 --save-dir outputs/resource_island_full
```

Cross-world validation and synthesis:

```bash
python run_known_answer_sanity_checks.py --save-dir outputs/known_answer_checks
python run_mechanism_traces.py --save-dir outputs/mechanism_traces
python run_claim_audit_suite.py --save-dir outputs/claim_audit_suite
python build_cross_world_synthesis.py --output-dir outputs/cross_world_synthesis
```

## Interactive Explorer

The `explorer/` app is a Vite/React interface for browsing selected cross-world results and the Pricing Arena price-cap audit.

```bash
cd explorer
npm install
npm run dev
```

Build for static hosting:

```bash
cd explorer
npm run build
```

The repository includes a GitHub Pages workflow for publishing the built explorer.

## Paper Build

The paper can be compiled from `paper/`:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Notes

This repository is the cleaned publication version of the project. The pre-publication working state is preserved in the local git branch:

```text
pre-publication-working-state-2026-07-28
```
