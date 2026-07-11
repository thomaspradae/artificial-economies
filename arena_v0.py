from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


MECHANISMS = (
    "none",
    "price_cap",
    "tax_high_price",
    "random_audit",
    "anti_collusion",
    "demand_shock",
)


@dataclass(frozen=True)
class MarketConfig:
    price_grid: np.ndarray = field(default_factory=lambda: np.linspace(1.0, 10.0, 19))
    cost: float = 1.0
    market_size: float = 100.0
    alpha: float = 0.9
    tau: float = 0.7
    quality: np.ndarray = field(default_factory=lambda: np.array([8.0, 8.0]))

    mechanism: str = "none"
    price_cap: float = 5.5
    tax_threshold: float = 5.5
    tax_rate: float = 0.30
    audit_probability: float = 0.08
    audit_threshold: float = 5.5
    audit_penalty: float = 35.0
    collusion_threshold: float = 5.5
    collusion_window: float = 0.75
    collusion_penalty: float = 20.0
    shock_probability: float = 0.03
    shock_scale: float = 0.30


def logsumexp(values: np.ndarray) -> float:
    top = float(np.max(values))
    return top + math.log(float(np.sum(np.exp(values - top))))


class DuopolyMarket:
    def __init__(self, config: MarketConfig, seed: int = 0):
        if config.mechanism not in MECHANISMS:
            valid = ", ".join(MECHANISMS)
            raise ValueError(f"Unknown mechanism {config.mechanism!r}; expected one of: {valid}")
        self.cfg = config
        self.rng = np.random.default_rng(seed)

    def demand(self, prices: np.ndarray, market_size: float | None = None) -> np.ndarray:
        """Logit demand over outside option plus both firms."""
        size = self.cfg.market_size if market_size is None else market_size
        utilities = self.cfg.quality - self.cfg.alpha * prices
        logits = np.concatenate(([0.0], utilities)) / self.cfg.tau
        probabilities = np.exp(logits - logsumexp(logits))
        return size * probabilities[1:]

    def consumer_surplus_proxy(self, prices: np.ndarray, market_size: float | None = None) -> float:
        """Inclusive-value surplus proxy. Good enough for comparing mechanisms in v0."""
        size = self.cfg.market_size if market_size is None else market_size
        utilities = self.cfg.quality - self.cfg.alpha * prices
        logits = np.concatenate(([0.0], utilities)) / self.cfg.tau
        return float(size * (self.cfg.tau / self.cfg.alpha) * logsumexp(logits))

    def step(self, action_indices: tuple[int, int]) -> tuple[tuple[int, int], np.ndarray, dict[str, float]]:
        raw_prices = self.cfg.price_grid[np.array(action_indices, dtype=int)]
        prices = raw_prices.copy()
        market_size = self.cfg.market_size
        audit_hit = 0.0

        if self.cfg.mechanism == "price_cap":
            prices = np.minimum(prices, self.cfg.price_cap)
        elif self.cfg.mechanism == "demand_shock":
            if self.rng.random() < self.cfg.shock_probability:
                multiplier = self.rng.lognormal(mean=0.0, sigma=self.cfg.shock_scale)
                market_size *= multiplier

        quantities = self.demand(prices, market_size=market_size)
        profits = (prices - self.cfg.cost) * quantities
        penalties = np.zeros(2)

        if self.cfg.mechanism == "tax_high_price":
            excess = np.maximum(prices - self.cfg.tax_threshold, 0.0)
            penalties = self.cfg.tax_rate * excess * quantities
        elif self.cfg.mechanism == "random_audit":
            both_high = np.mean(prices) >= self.cfg.audit_threshold
            if both_high and self.rng.random() < self.cfg.audit_probability:
                penalties[:] = self.cfg.audit_penalty
                audit_hit = 1.0
        elif self.cfg.mechanism == "anti_collusion":
            both_high = np.mean(prices) >= self.cfg.collusion_threshold
            close_prices = abs(float(prices[0] - prices[1])) <= self.cfg.collusion_window
            if both_high and close_prices:
                penalties[:] = self.cfg.collusion_penalty

        rewards = profits - penalties
        consumer_surplus = self.consumer_surplus_proxy(prices, market_size=market_size)
        welfare = float(np.sum(profits) + consumer_surplus)
        next_state = tuple(int(i) for i in action_indices)

        info = {
            "p1": float(prices[0]),
            "p2": float(prices[1]),
            "raw_p1": float(raw_prices[0]),
            "raw_p2": float(raw_prices[1]),
            "avg_price": float(np.mean(prices)),
            "quantity1": float(quantities[0]),
            "quantity2": float(quantities[1]),
            "profit1": float(profits[0]),
            "profit2": float(profits[1]),
            "reward1": float(rewards[0]),
            "reward2": float(rewards[1]),
            "consumer_surplus": consumer_surplus,
            "welfare": welfare,
            "penalty1": float(penalties[0]),
            "penalty2": float(penalties[1]),
            "market_size": float(market_size),
            "audit_hit": audit_hit,
        }
        return next_state, rewards, info


class QAgent:
    def __init__(self, n_prices: int, lr: float = 0.08, gamma: float = 0.96, seed: int = 0):
        self.n_prices = n_prices
        self.lr = lr
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.q_values = np.zeros((n_prices, n_prices, n_prices))

    def greedy_action(self, state: tuple[int, int]) -> int:
        return int(np.argmax(self.q_values[state[0], state[1], :]))

    def act(self, state: tuple[int, int], epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_prices))
        return self.greedy_action(state)

    def update(self, state: tuple[int, int], action: int, reward: float, next_state: tuple[int, int]) -> None:
        old_value = self.q_values[state[0], state[1], action]
        future_value = float(np.max(self.q_values[next_state[0], next_state[1], :]))
        target = reward + self.gamma * future_value
        self.q_values[state[0], state[1], action] += self.lr * (target - old_value)


@dataclass
class TrainingRun:
    data: dict[str, np.ndarray]
    benchmarks: dict[str, object]
    agent1: QAgent
    agent2: QAgent
    config: MarketConfig
    final_state: tuple[int, int]
    final_epsilon: float


def compute_static_benchmarks(price_grid: np.ndarray) -> dict[str, object]:
    """Grid-search one-shot Nash pairs and symmetric joint-profit price."""
    cfg = MarketConfig(price_grid=price_grid, mechanism="none")
    env = DuopolyMarket(cfg)
    n_prices = len(price_grid)
    pay1 = np.zeros((n_prices, n_prices))
    pay2 = np.zeros((n_prices, n_prices))

    for i in range(n_prices):
        for j in range(n_prices):
            _, _, info = env.step((i, j))
            pay1[i, j] = info["profit1"]
            pay2[i, j] = info["profit2"]

    nash_pairs: list[tuple[float, float]] = []
    for i in range(n_prices):
        for j in range(n_prices):
            firm1_best = pay1[i, j] >= np.max(pay1[:, j]) - 1e-9
            firm2_best = pay2[i, j] >= np.max(pay2[i, :]) - 1e-9
            if firm1_best and firm2_best:
                nash_pairs.append((float(price_grid[i]), float(price_grid[j])))

    joint_profit_diag = []
    for i in range(n_prices):
        _, _, info = env.step((i, i))
        joint_profit_diag.append(info["profit1"] + info["profit2"])

    monopoly_price = float(price_grid[int(np.argmax(joint_profit_diag))])
    nash_price = float(np.mean(nash_pairs[0])) if nash_pairs else None
    return {
        "nash_pairs": nash_pairs,
        "nash_price": nash_price,
        "monopoly_price": monopoly_price,
    }


def collusion_index(avg_price: float, nash_price: float | None, monopoly_price: float) -> float:
    if nash_price is None or abs(monopoly_price - nash_price) < 1e-12:
        return float("nan")
    scaled = (avg_price - nash_price) / (monopoly_price - nash_price)
    return float(np.clip(scaled, 0.0, 1.0))


def _add_learning_metrics(
    info: dict[str, float],
    step: int,
    epsilon: float,
    benchmarks: dict[str, object],
) -> None:
    info["step"] = float(step)
    info["epsilon"] = float(epsilon)
    info["collusion_index"] = collusion_index(
        info["avg_price"],
        benchmarks["nash_price"],
        float(benchmarks["monopoly_price"]),
    )


def records_to_arrays(records: list[dict[str, float]]) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("cannot convert empty records to arrays")
    return {key: np.array([record[key] for record in records]) for key in records[0].keys()}


def train_market_with_agents(
    mechanism: str = "none",
    steps: int = 40_000,
    seed: int = 7,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.03,
    epsilon_decay: float = 0.99985,
) -> TrainingRun:
    if steps < 1:
        raise ValueError("steps must be positive")

    cfg = MarketConfig(mechanism=mechanism)
    env = DuopolyMarket(cfg, seed=seed)
    n_prices = len(cfg.price_grid)
    agent1 = QAgent(n_prices, seed=seed)
    agent2 = QAgent(n_prices, seed=seed + 1)
    benchmarks = compute_static_benchmarks(cfg.price_grid)

    state = (n_prices // 2, n_prices // 2)
    epsilon = epsilon_start
    records: list[dict[str, float]] = []

    for t in range(steps):
        a1 = agent1.act(state, epsilon)
        a2 = agent2.act(state, epsilon)
        next_state, rewards, info = env.step((a1, a2))
        agent1.update(state, a1, float(rewards[0]), next_state)
        agent2.update(state, a2, float(rewards[1]), next_state)

        state = next_state
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        _add_learning_metrics(info, t, epsilon, benchmarks)
        records.append(info)

    return TrainingRun(
        data=records_to_arrays(records),
        benchmarks=benchmarks,
        agent1=agent1,
        agent2=agent2,
        config=cfg,
        final_state=state,
        final_epsilon=epsilon,
    )


def train_market(
    mechanism: str = "none",
    steps: int = 40_000,
    seed: int = 7,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.03,
    epsilon_decay: float = 0.99985,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    run = train_market_with_agents(
        mechanism=mechanism,
        steps=steps,
        seed=seed,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )
    return run.data, run.benchmarks


def evaluate_policy_pair(
    agent1: QAgent,
    agent2: QAgent,
    mechanism: str = "none",
    steps: int = 5_000,
    seed: int = 10_000,
    initial_state: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    if steps < 1:
        raise ValueError("steps must be positive")

    cfg = MarketConfig(mechanism=mechanism)
    env = DuopolyMarket(cfg, seed=seed)
    n_prices = len(cfg.price_grid)
    benchmarks = compute_static_benchmarks(cfg.price_grid)
    state = initial_state if initial_state is not None else (n_prices // 2, n_prices // 2)
    records: list[dict[str, float]] = []

    for t in range(steps):
        a1 = agent1.greedy_action(state)
        a2 = agent2.greedy_action(state)
        next_state, _, info = env.step((a1, a2))
        state = next_state
        _add_learning_metrics(info, t, 0.0, benchmarks)
        records.append(info)

    return records_to_arrays(records)


def train_adversary_against_frozen_firm1(
    frozen_agent1: QAgent,
    mechanism: str = "none",
    steps: int = 20_000,
    seed: int = 20_000,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.03,
    epsilon_decay: float = 0.99985,
    initial_state: tuple[int, int] | None = None,
) -> TrainingRun:
    if steps < 1:
        raise ValueError("steps must be positive")

    cfg = MarketConfig(mechanism=mechanism)
    env = DuopolyMarket(cfg, seed=seed)
    n_prices = len(cfg.price_grid)
    adversary = QAgent(n_prices, seed=seed)
    benchmarks = compute_static_benchmarks(cfg.price_grid)
    state = initial_state if initial_state is not None else (n_prices // 2, n_prices // 2)
    epsilon = epsilon_start
    records: list[dict[str, float]] = []

    for t in range(steps):
        a1 = frozen_agent1.greedy_action(state)
        a2 = adversary.act(state, epsilon)
        next_state, rewards, info = env.step((a1, a2))
        adversary.update(state, a2, float(rewards[1]), next_state)

        state = next_state
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        _add_learning_metrics(info, t, epsilon, benchmarks)
        records.append(info)

    return TrainingRun(
        data=records_to_arrays(records),
        benchmarks=benchmarks,
        agent1=frozen_agent1,
        agent2=adversary,
        config=cfg,
        final_state=state,
        final_epsilon=epsilon,
    )


def rolling_mean(values: np.ndarray, window: int = 500) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def summarize_run(data: dict[str, np.ndarray], tail: int = 1000) -> dict[str, float]:
    tail = min(tail, len(next(iter(data.values()))))
    sl = slice(-tail, None)
    return {
        "avg_price": float(np.mean(data["avg_price"][sl])),
        "profit_total": float(np.mean(data["profit1"][sl] + data["profit2"][sl])),
        "consumer_surplus": float(np.mean(data["consumer_surplus"][sl])),
        "welfare": float(np.mean(data["welfare"][sl])),
        "collusion_index": float(np.nanmean(data["collusion_index"][sl])),
        "penalty_total": float(np.mean(data["penalty1"][sl] + data["penalty2"][sl])),
    }


def write_trace_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(data.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row_idx in range(len(data[keys[0]])):
            writer.writerow({key: data[key][row_idx] for key in keys})


def write_summary_csv(path: Path, summaries: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["mechanism", *next(iter(summaries.values())).keys()]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics)
        writer.writeheader()
        for mechanism, summary in summaries.items():
            writer.writerow({"mechanism": mechanism, **summary})


def plot_comparison(
    results: dict[str, dict[str, np.ndarray]],
    benchmarks: dict[str, object],
    save_dir: Path,
    rolling_window: int = 500,
    show: bool = False,
) -> list[Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(save_dir / ".matplotlib"))

    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved_paths: list[Path] = []

    plot_specs = [
        ("avg_price", "Average Price", "average_price.png"),
        ("welfare", "Welfare Proxy", "welfare.png"),
        ("consumer_surplus", "Consumer Surplus Proxy", "consumer_surplus.png"),
        ("collusion_index", "Collusion Index", "collusion_index.png"),
    ]

    for metric, title, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(11, 6))
        for mechanism, data in results.items():
            series = rolling_mean(data[metric], rolling_window)
            ax.plot(series, label=mechanism)

        if metric == "avg_price":
            nash_price = benchmarks["nash_price"]
            monopoly_price = float(benchmarks["monopoly_price"])
            if nash_price is not None:
                ax.axhline(float(nash_price), linestyle="--", linewidth=1.2, label=f"Nash price ~ {nash_price:.2f}")
            ax.axhline(monopoly_price, linestyle=":", linewidth=1.2, label=f"Joint-profit price ~ {monopoly_price:.2f}")
        elif metric == "collusion_index":
            ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.35)
            ax.axhline(1.0, linestyle=":", linewidth=1.0, color="black", alpha=0.35)

        ax.set_title(title)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()

        output_path = save_dir / filename
        fig.savefig(output_path, dpi=160)
        saved_paths.append(output_path)
        if show:
            plt.show()
        plt.close(fig)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two Q-learning firms in a repeated pricing market.")
    parser.add_argument("--steps", type=int, default=40_000, help="Training steps per mechanism.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs"), help="Directory for CSV and PNG outputs.")
    parser.add_argument("--rolling-window", type=int, default=500, help="Smoothing window for plots.")
    parser.add_argument("--show", action="store_true", help="Show Matplotlib windows after saving plots.")
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        choices=MECHANISMS,
        default=list(MECHANISMS),
        help="Mechanisms to compare.",
    )
    return parser.parse_args()


def print_summary(mechanism: str, summary: dict[str, float]) -> None:
    print(f"{mechanism:16s} avg_price={summary['avg_price']:.2f} "
          f"welfare={summary['welfare']:.2f} "
          f"collusion={summary['collusion_index']:.2f} "
          f"penalty={summary['penalty_total']:.2f}")


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    results: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, float]] = {}
    final_benchmarks: dict[str, object] | None = None

    for offset, mechanism in enumerate(args.mechanisms):
        print(f"Training mechanism: {mechanism}")
        data, benchmarks = train_market(mechanism=mechanism, steps=args.steps, seed=args.seed + offset)
        results[mechanism] = data
        summaries[mechanism] = summarize_run(data)
        final_benchmarks = benchmarks
        write_trace_csv(args.save_dir / f"{mechanism}_trace.csv", data)
        print_summary(mechanism, summaries[mechanism])

    assert final_benchmarks is not None
    print()
    print("Static one-shot Nash price pairs:", final_benchmarks["nash_pairs"])
    print("Symmetric joint-profit price:", final_benchmarks["monopoly_price"])

    write_summary_csv(args.save_dir / "summary.csv", summaries)
    try:
        saved_plots = plot_comparison(results, final_benchmarks, args.save_dir, args.rolling_window, args.show)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        saved_plots = []
        print()
        print("Skipping plots because matplotlib is not installed. Run: pip install -r requirements.txt")

    print()
    print(f"Wrote summary: {args.save_dir / 'summary.csv'}")
    for path in saved_plots:
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
