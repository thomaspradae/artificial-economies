from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def new_run_manifest(config: dict[str, Any], out_dir: str | Path) -> str:
    """Write a JSON manifest with config, git commit hash, and timestamp."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = str(config.get("run_id", timestamp))
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "config": config,
    }
    (output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return run_id


def log_step(run_id: str, step_idx: int, metrics: dict[str, Any], out_dir: str | Path = ".") -> None:
    """Append one metrics row to the run CSV."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_id}_steps.csv"
    row = {"step_idx": step_idx, **metrics}
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def finalize_run(run_id: str, summary_metrics: dict[str, Any], out_dir: str | Path = ".") -> None:
    """Write final summary JSON for the run."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_metrics": summary_metrics,
    }
    (output_dir / f"{run_id}_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
