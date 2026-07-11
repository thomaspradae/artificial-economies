from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from arena_v0 import MarketConfig
from run_multiseed import finite_mean, summarize_final_window
from worlds.pricing_arena.env import PricingArenaWorld
from minds.deep_rl.dqn_mind import DQNMind
from minds.deep_rl.ppo_mind import PPOMind
from worlds.pricing_arena.training import SUPPORTED_MINDS, train_market


BEST_RESPONSE_MINDS = ("dqn", "ppo", "independent_dqn")
PHASE3_MINDS = ("dqn", "ppo", "independent_dqn", "centralized_critic")
COMPARISON_PAIRS = (("dqn", "simple_dqn"), ("ppo", "simple_ppo"))


def true_best_response(opponent_action: int, mechanism: str = "none", seed: int = 0) -> dict[str, Any]:
    """Compute firm-1 one-step best response against a fixed firm-2 action."""
    cfg = MarketConfig(mechanism=mechanism)
    rows = []
    for action in range(len(cfg.price_grid)):
        world = PricingArenaWorld(config=cfg, seed=seed)
        _, _, _, info = world.step([action, opponent_action])
        rows.append({"action": action, "profit": info["profit1"], "price": info["p1"]})
    best = max(rows, key=lambda row: row["profit"])
    return {
        "best_action": int(best["action"]),
        "best_price": float(best["price"]),
        "best_profit": float(best["profit"]),
        "all_rows": rows,
    }


def make_mind(mind: str, seed: int):
    """Mind configuration for the one-step best-response validation."""
    if mind in {"dqn", "independent_dqn"}:
        return DQNMind(
            action_dim=19,
            obs_dim=38,
            hidden_dim=32,
            lr=1e-3,
            gamma=0.0,
            epsilon_decay=0.999,
            batch_size=16,
            min_replay_size=16,
            target_update_interval=50,
            seed=seed,
        )
    if mind == "ppo":
        return PPOMind(
            action_dim=19,
            obs_dim=38,
            hidden_dim=32,
            gamma=0.0,
            rollout_steps=16,
            batch_size=16,
            policy_lr=1e-2,
            value_lr=1e-3,
            entropy_coef=0.01,
            seed=seed,
        )
    raise ValueError(f"unsupported best-response mind {mind!r}")


def validate_best_response(
    mind: str,
    steps: int,
    seed: int,
    opponent_action: int,
    mechanism: str = "none",
) -> dict[str, Any]:
    if mind not in BEST_RESPONSE_MINDS:
        raise ValueError(f"{mind!r} is not a single-agent best-response validation mind")
    cfg = MarketConfig(mechanism=mechanism)
    world = PricingArenaWorld(config=cfg, seed=seed)
    agent = make_mind(mind, seed)
    state = world.state

    for _ in range(steps):
        action = int(agent.act(state))
        next_state, rewards, _, _ = world.step([action, opponent_action])
        agent.update(state, action, float(rewards[0]), next_state, False)
        state = next_state

    learned_action = int(agent.greedy_action(state)) if hasattr(agent, "greedy_action") else int(agent.act(state))
    probe = PricingArenaWorld(config=cfg, seed=seed + 1)
    _, _, _, learned_info = probe.step([learned_action, opponent_action])
    optimum = true_best_response(opponent_action, mechanism=mechanism, seed=seed + 2)
    regret = float(optimum["best_profit"] - learned_info["profit1"])
    return {
        "mind": mind,
        "seed": seed,
        "steps": steps,
        "mechanism": mechanism,
        "opponent_action": opponent_action,
        "learned_action": learned_action,
        "learned_price": learned_info["p1"],
        "learned_profit": learned_info["profit1"],
        "best_action": optimum["best_action"],
        "best_price": optimum["best_price"],
        "best_profit": optimum["best_profit"],
        "regret": regret,
        "matched_best_action": learned_action == optimum["best_action"],
    }


def validate_multimind_smoke(mind: str, steps: int, seed: int, mechanism: str) -> dict[str, Any]:
    data, _ = train_market(mechanism=mechanism, steps=steps, seed=seed, mind=mind)
    return {
        "mind": mind,
        "seed": seed,
        "steps": steps,
        "mechanism": mechanism,
        "avg_price": finite_mean(data["avg_price"]),
        "welfare": finite_mean(data["welfare"]),
        "collusion_index": finite_mean(data["collusion_index"]),
        "has_nan": bool(
            np.any(~np.isfinite(data["avg_price"]))
            or np.any(~np.isfinite(data["welfare"]))
            or np.any(~np.isfinite(data["collusion_index"]))
        ),
    }


def qualitative_torch_simple_comparison(
    torch_mind: str,
    simple_mind: str,
    steps: int,
    final_window: int,
    seed: int,
    mechanisms: tuple[str, ...] = ("none", "price_cap"),
) -> list[dict[str, Any]]:
    """Compare torch and NumPy versions on matched seeds/mechanisms."""
    rows: list[dict[str, Any]] = []
    for mechanism in mechanisms:
        for implementation, mind in (("torch", torch_mind), ("numpy_simple", simple_mind)):
            data, benchmarks = train_market(mechanism=mechanism, steps=steps, seed=seed, mind=mind)
            summary = summarize_final_window(data, benchmarks, final_window)
            rows.append(
                {
                    "family": torch_mind,
                    "implementation": implementation,
                    "mind": mind,
                    "seed": seed,
                    "steps": steps,
                    "final_window": final_window,
                    "mechanism": mechanism,
                    "avg_price": summary["avg_price"],
                    "welfare": summary["welfare"],
                    "collusion_index": summary["collusion_index"],
                    "has_nan": not bool(
                        np.isfinite(summary["avg_price"])
                        and np.isfinite(summary["welfare"])
                        and np.isfinite(summary["collusion_index"])
                    ),
                }
            )
    return rows


def add_suppression_flags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark whether price_cap lowers collusion relative to none per implementation."""
    indexed = {
        (row["family"], row["implementation"], row["mechanism"]): row
        for row in rows
    }
    for row in rows:
        none = indexed.get((row["family"], row["implementation"], "none"))
        cap = indexed.get((row["family"], row["implementation"], "price_cap"))
        if none is None or cap is None:
            row["price_cap_lowers_collusion"] = ""
            continue
        row["price_cap_lowers_collusion"] = bool(
            float(cap["collusion_index"]) < float(none["collusion_index"])
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_validation(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    best_response_rows = [
        validate_best_response(mind, args.best_response_steps, args.seed, args.opponent_action, args.mechanism)
        for mind in args.minds
        if mind in BEST_RESPONSE_MINDS
    ]
    smoke_rows = [
        validate_multimind_smoke(mind, args.smoke_steps, args.seed, args.mechanism)
        for mind in args.minds
    ]
    comparison_rows: list[dict[str, Any]] = []
    if args.comparison_steps > 0:
        for torch_mind, simple_mind in COMPARISON_PAIRS:
            comparison_rows.extend(
                qualitative_torch_simple_comparison(
                    torch_mind,
                    simple_mind,
                    args.comparison_steps,
                    args.comparison_final_window,
                    args.seed,
                )
            )
        add_suppression_flags(comparison_rows)

    best_response_path = args.save_dir / "best_response_validation.csv"
    smoke_path = args.save_dir / "pricing_smoke_validation.csv"
    comparison_path = args.save_dir / "torch_vs_numpy_qualitative.csv"
    write_csv(best_response_path, best_response_rows)
    write_csv(smoke_path, smoke_rows)
    write_csv(comparison_path, comparison_rows)

    manifest_path = args.save_dir / "experiment_manifest.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - start,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "notes": {
            "best_response": "firm 1 learns against a fixed firm-2 action; regret is optimal one-step profit minus learned-action profit",
            "smoke": "multi-agent Pricing Arena run must emit finite metrics for each Phase 3 mind",
            "torch_vs_numpy": "matched-seed smoke comparison between torch minds and explicit simple NumPy baselines; price_cap_lowers_collusion tests qualitative direction only",
        },
        "outputs": [str(best_response_path), str(smoke_path), str(comparison_path)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "best_response": best_response_path,
        "pricing_smoke": smoke_path,
        "torch_vs_numpy": comparison_path,
        "manifest": manifest_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 3 neural/MARL minds on Pricing Arena.")
    parser.add_argument("--minds", nargs="+", choices=PHASE3_MINDS, default=list(PHASE3_MINDS))
    parser.add_argument("--best-response-steps", type=int, default=5_000)
    parser.add_argument("--smoke-steps", type=int, default=500)
    parser.add_argument("--comparison-steps", type=int, default=1_000)
    parser.add_argument("--comparison-final-window", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mechanism", choices=("none", "price_cap"), default="none")
    parser.add_argument("--opponent-action", type=int, default=9)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/phase3_validation"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    outputs = run_validation(parse_args(argv))
    for output in outputs.values():
        print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
