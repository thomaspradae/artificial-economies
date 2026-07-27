# Foundation Papers Summary

## What Was Built

A simplified theory-foundation layer now exists for the thesis. It does not try to rank hundreds of papers. It keeps a curated list of the papers that should ground each world and the shared learning-agent methods.

Generated files:

- `literature/foundation_papers.md`
- `literature/foundation_papers.csv`
- `literature/foundation_pdf_queue.csv`

Regenerate them with:

```bash
python -m tools.theory_scout.cli foundation-papers
```

## Current Status

| Area | Foundation papers | Found in cache | Ready for LLM | Missing text |
| --- | ---: | ---: | ---: | ---: |
| Cross-world methods | 5 | 0 | 0 | 5 |
| Pricing Arena | 6 | 5 | 1 | 5 |
| Auction House | 5 | 2 | 2 | 3 |
| Public Goods | 6 | 3 | 1 | 5 |
| Resource Island | 6 | 1 | 1 | 5 |
| Labor Market | 5 | 2 | 2 | 3 |

Overall:

- `33` curated foundation papers.
- `7` already have extracted text and are ready for LLM reading.
- `26` need PDF/text.
- `20` are important canonical anchors that the broad metadata scout did not catch cleanly.

## Ready For LLM Now

These already have extracted text under `literature/text/`:

- Calvano et al. (2020), *Artificial Intelligence, Algorithmic Pricing, and Collusion*.
- Dütting et al., *Optimal auctions through deep learning* / related survey text.
- Nédelec et al. (2020), *Learning in repeated auctions*.
- Sefton, Shupp, and Walker (2007), *The Effect of Rewards and Sanctions in Provision of Public Goods*.
- Cox, Arnold, and Villamayor-Tomas (2010), *A Review of Design Principles for Community-based Natural Resource Management*.
- Abdulkadiroglu, Pathak, Roth, and Sonmez (2006), *Changing the Boston School Choice Mechanism*.
- Roth (2007), *Deferred Acceptance Algorithms: History, Theory, Practice, and Open Questions*.

## Highest-Priority Manual PDF/Text Queue

Start with the `must_read` rows in:

```text
literature/foundation_pdf_queue.csv
```

Key missing anchors:

- Watkins and Dayan (1992), *Q-learning*.
- Mnih et al. (2015), *Human-level control through deep reinforcement learning*.
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*.
- Tan (1993), *Multi-Agent Reinforcement Learning: Independent versus Cooperative Agents*.
- Lowe et al. (2017), *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*.
- Tirole (1988), *The Theory of Industrial Organization*.
- Calvano et al. (2021), *Algorithmic collusion with imperfect monitoring*.
- Vickrey (1961), *Counterspeculation, Auctions, and Competitive Sealed Tenders*.
- Myerson (1981), *Optimal Auction Design*.
- Samuelson (1954), *The Pure Theory of Public Expenditure*.
- Olson (1965), *The Logic of Collective Action*.
- Fehr and Gachter (2000), *Cooperation and Punishment in Public Goods Experiments*.
- Hardin (1968), *The Tragedy of the Commons*.
- Ostrom (1990), *Governing the Commons*.
- Gale and Shapley (1962), *College Admissions and the Stability of Marriage*.
- Roth (1982), *The Economics of Matching: Stability and Incentives*.

## Why This Matters

This file converts the literature task into a concrete checklist:

- Each world has a classical prediction.
- Each world has a benchmark obligation.
- Each learning mind has a canonical method anchor.
- Each institution has a theory-facing metric obligation.
- Missing papers are explicitly marked instead of hidden inside noisy search results.

## Validation

The implementation was tested with:

```bash
.venv/bin/python -m unittest test_theory_scout.py
.venv/bin/python -m unittest discover
```

Observed result:

- Focused theory scout tests: `20` passed.
- Full test discovery: `148` passed.

