# Labor Market Design Lock

Labor Market is a small two-sided matching world. Its purpose is to test
strategic manipulation and stability under a benchmark mechanism rather than to
simulate all labor-market frictions.

## Agents

- Learning agents are workers.
- Employers are non-learning market participants with fixed preferences over
  workers.
- Each worker has a true preference ranking over employers.
- Each employer has a true preference ranking over workers.

## Observation

Tabular Q-learning receives:

```text
(worker_bin, top_choice_bin)
```

- `worker_bin` identifies the worker modulo the action count.
- `top_choice_bin` is the worker's true top employer.

This keeps the shared `QLearningMind` update rule unchanged while giving the
learner enough information to decide whether to report truthfully or manipulate.

## Actions

Each worker action is a reported top employer. The world builds a full reported
preference list by putting that employer first and appending the remaining true
preference order. This is intentionally smaller than a full permutation action
space, but it creates a real manipulation channel.

## Matching Protocol

The v0 institution is worker-proposing deferred acceptance:

1. unmatched workers propose to the next employer in their reported list;
2. each employer tentatively keeps its most preferred proposer/current match;
3. rejected workers continue until no unmatched worker has proposals left.

Tie-breaking is deterministic by the employer preference order. The benchmark
uses truthful worker reports.

## Rewards

- A matched worker receives its true utility for the assigned employer.
- An unmatched worker receives zero.
- Employer welfare is recorded but employers do not learn in v0.

## Metrics

- `match_rate`
- `worker_welfare`
- `employer_welfare`
- `total_welfare`
- `blocking_pairs`
- `stability`
- `truthful_report_rate`
- `manipulation_gain_mean`

## Validation Bar

- deferred-acceptance mechanics must pass hand-computed tests;
- stability and blocking-pair checks must be tested;
- Q-learning workers must run unchanged;
- smoke output must report whether workers learn truthful or manipulative
  reports relative to the truthful DA benchmark.
