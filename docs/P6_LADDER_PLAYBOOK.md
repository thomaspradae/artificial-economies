# P.6 Ladder Wiring - Reusable Playbook

For any world `W` with an existing tabular Q-learning implementation:

1. Audit whether `W` exposes a structured non-tabular observation for deep RL.
   If not, build `worlds/W/features.py::structured_observations()` as a fixed-width
   vector encoder. Encoders should be independent of transient Python objects,
   use finite numeric values, and normalize positions, quantities, and ranks to
   stable `[0, 1]`-scale ranges where possible.
2. Confirm the action space is discrete and compatible with categorical policy
   and Q-value heads. Existing tabular worlds should already satisfy this.
3. Wire `worlds/W/training.py` to accept `q_learning`, `dqn`, `ppo`,
   `independent_dqn`, and `centralized_critic` through the shared
   `worlds.mind_ladder` helper.
4. Sanity check the minimal world case before adding institutions. Examples:
   fixed-value auction, low-agent commons case, or fixed-preference matching case.
5. Qualitative check tabular Q versus DQN on the baseline institution before
   treating neural runs as evidence.
6. Smoke run all supported minds over the world institutions with short steps and
   `n_seeds=2-3`. Confirm CSVs write, all rows are finite, and every institution
   appears for every mind.
7. Full run all four neural/MARL minds with the same step count and seed count as
   the tabular full run. Distribute long jobs by machine, one heavy job per small
   host unless memory checks show room.
8. Validate row counts, seed coverage, and absence of NaN/inf before interpreting
   results.
9. Build `outputs/<world>_phase3_full/mind_comparison.csv` with
   `build_world_mind_comparison.py`, not a per-world copy.
10. Write the world-specific capability finding into `ROADMAP_STATUS.md` in prose
    immediately. A CSV alone is not a research result.

Stop conditions:

- A world is only P.6-complete when the full comparison table exists, validates,
  and has an interpreted capability result.
- If a mind cannot be wired without changing the shared `Agent` interface, treat
  that as an abstraction bug and document the interface fix.
