from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from worlds.resource_island.env import FOOD, ResourceIslandConfig
from worlds.resource_island.training import summarize_records, train_resource_island


def _resource_patch(grid_size: int) -> Any:
    import numpy as np

    resources = np.zeros((grid_size, grid_size, 2), dtype=int)
    resources[0, 0, FOOD] = 3
    resources[1, 0, FOOD] = 2
    return resources


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    single_rows: list[dict[str, Any]] = []
    for mind in ("dqn", "ppo"):
        cfg = ResourceIslandConfig(
            grid_size=3,
            n_agents=1,
            max_steps=args.single_agent_steps,
            start_positions=((0, 0),),
            initial_resource_units=0,
            initial_resources=_resource_patch(3),
            resource_spawn_probability=0.0,
        )
        result = train_resource_island(
            steps=args.single_agent_steps,
            seed=args.seed,
            config=cfg,
            mind=mind,
            obs_radius=args.obs_radius,
        )
        summary = summarize_records(result.records, final_window=args.final_window)
        single_rows.append(
            {
                "mind": mind,
                "seed": args.seed,
                "steps": args.single_agent_steps,
                "obs_dim": result.obs_dim,
                **summary,
            }
        )

    comparison_rows: list[dict[str, Any]] = []
    cfg = ResourceIslandConfig(
        grid_size=4,
        n_agents=2,
        max_steps=args.comparison_steps,
        initial_resource_units=8,
        resource_spawn_probability=0.04,
    )
    for mind in ("q_learning", "dqn"):
        result = train_resource_island(
            steps=args.comparison_steps,
            seed=args.seed,
            config=cfg,
            mind=mind,
            obs_radius=args.obs_radius,
        )
        summary = summarize_records(result.records, final_window=args.final_window)
        comparison_rows.append(
            {
                "mind": mind,
                "seed": args.seed,
                "steps": args.comparison_steps,
                "obs_dim": result.obs_dim,
                **summary,
            }
        )

    single_path = args.save_dir / "single_agent_validation.csv"
    comparison_path = args.save_dir / "qualitative_comparison.csv"
    manifest_path = args.save_dir / "experiment_manifest.json"
    fieldnames = [
        "mind",
        "seed",
        "steps",
        "obs_dim",
        "survival_rate",
        "welfare",
        "gathered_food",
        "gathered_wood",
        "trade_attempt_count",
        "trade_count",
        "resource_sustainability",
    ]
    write_csv(single_path, single_rows, fieldnames)
    write_csv(comparison_path, comparison_rows, fieldnames)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - start,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "world": "resource_island",
        "purpose": "Phase 3 Resource Island cross-mind validation",
        "config": {
            "seed": args.seed,
            "single_agent_steps": args.single_agent_steps,
            "comparison_steps": args.comparison_steps,
            "final_window": args.final_window,
            "obs_radius": args.obs_radius,
        },
        "outputs": [str(single_path), str(comparison_path)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "single_agent_validation": single_path,
        "qualitative_comparison": comparison_path,
        "manifest": manifest_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate neural/MARL Resource Island wiring.")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/resource_island_phase3_validation"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--single-agent-steps", type=int, default=150)
    parser.add_argument("--comparison-steps", type=int, default=200)
    parser.add_argument("--final-window", type=int, default=50)
    parser.add_argument("--obs-radius", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    outputs = run(parse_args(argv))
    for path in outputs.values():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
