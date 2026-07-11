from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE3_MINDS = ("dqn", "ppo", "independent_dqn", "centralized_critic")


@dataclass(frozen=True)
class JobSpec:
    name: str
    command: list[str]
    required_outputs: tuple[Path, ...]


@dataclass
class JobState:
    spec: JobSpec
    state: str = "pending"
    pid: int | None = None
    returncode: int | None = None
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    elapsed_seconds: float | None = None
    log_path: Path | None = None
    process: subprocess.Popen[bytes] | None = field(default=None, repr=False)
    started_monotonic: float | None = field(default=None, repr=False)

    def status_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "pid": self.pid,
            "returncode": self.returncode,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "log_path": str(self.log_path) if self.log_path is not None else None,
            "command": self.spec.command,
            "required_outputs": [str(path) for path in self.spec.required_outputs],
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def phase3_jobs(python: str) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    for mind in PHASE3_MINDS:
        output = Path(f"outputs/{mind}_v0_multiseed")
        jobs.append(
            JobSpec(
                name=f"{mind}_multiseed",
                command=[
                    python,
                    "run_multiseed.py",
                    "--mind",
                    mind,
                    "--steps",
                    "40000",
                    "--n-seeds",
                    "20",
                    "--save-dir",
                    str(output),
                ],
                required_outputs=(
                    output / "summary_by_seed.csv",
                    output / "summary_aggregate.csv",
                    output / "mechanism_rankings.csv",
                    output / "experiment_manifest.json",
                ),
            )
        )

    for mind in PHASE3_MINDS:
        output = Path(f"outputs/{mind}_v1_exploitability")
        jobs.append(
            JobSpec(
                name=f"{mind}_exploitability",
                command=[
                    python,
                    "run_exploitability.py",
                    "--incumbent-mind",
                    mind,
                    "--save-dir",
                    str(output),
                ],
                required_outputs=(
                    output / "restarts_by_seed.csv",
                    output / "summary_by_seed.csv",
                    output / "summary_aggregate.csv",
                    output / "experiment_manifest.json",
                ),
            )
        )
    return jobs


def output_complete(spec: JobSpec) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in spec.required_outputs)


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(f"\n[{utc_now()}] START {' '.join(command)}\n")
        handle.flush()
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, check=False)
        handle.write(f"[{utc_now()}] END returncode={completed.returncode}\n")
    return int(completed.returncode)


def write_status(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    tmp.replace(path)


def status_payload(args: argparse.Namespace, jobs: dict[str, JobState], overall_state: str) -> dict[str, Any]:
    return {
        "created_at_utc": args.created_at_utc,
        "updated_at_utc": utc_now(),
        "overall_state": overall_state,
        "max_parallel": args.max_parallel,
        "torch_threads": args.torch_threads,
        "python": args.python,
        "jobs": {name: job.status_dict() for name, job in jobs.items()},
        "postprocess": {
            "combined_table": str(args.combined_output),
            "validation_report": str(args.validation_report),
        },
    }


def managed_env(torch_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    thread_value = str(torch_threads)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = thread_value
    env["TORCH_NUM_THREADS"] = thread_value
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    return env


def start_job(job: JobState, env: dict[str, str]) -> None:
    assert job.log_path is not None
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    with job.log_path.open("a") as handle:
        handle.write(f"\n[{utc_now()}] START {' '.join(job.spec.command)}\n")
        handle.flush()
        process = subprocess.Popen(job.spec.command, stdout=handle, stderr=subprocess.STDOUT, env=env)

    job.process = process
    job.pid = process.pid
    job.state = "running"
    job.started_at_utc = utc_now()
    job.started_monotonic = time.monotonic()


def update_running_jobs(jobs: dict[str, JobState]) -> None:
    for job in jobs.values():
        if job.state != "running" or job.process is None:
            continue
        returncode = job.process.poll()
        if returncode is None:
            continue
        job.returncode = int(returncode)
        job.finished_at_utc = utc_now()
        if job.started_monotonic is not None:
            job.elapsed_seconds = time.monotonic() - job.started_monotonic
        job.state = "complete" if returncode == 0 and output_complete(job.spec) else "failed"


def pending_jobs(jobs: dict[str, JobState]) -> list[JobState]:
    return [job for job in jobs.values() if job.state == "pending"]


def running_jobs(jobs: dict[str, JobState]) -> list[JobState]:
    return [job for job in jobs.values() if job.state == "running"]


def all_done(jobs: dict[str, JobState]) -> bool:
    return all(job.state in {"skipped", "complete", "failed"} for job in jobs.values())


def any_failed(jobs: dict[str, JobState]) -> bool:
    return any(job.state == "failed" for job in jobs.values())


def build_combined_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        "build_combined_table.py",
        "--multiseed-dir",
        "outputs/full_v0_multiseed",
        "--exploitability-dir",
        "outputs/v1_exploitability",
        "--mind",
        "q_learning",
        "--random-multiseed-dir",
        "outputs/random_v0_multiseed",
        "--random-exploitability-dir",
        "outputs/random_v1_exploitability",
    ]
    for mind in PHASE3_MINDS:
        command.extend(
            [
                "--result",
                f"{mind}:outputs/{mind}_v0_multiseed:outputs/{mind}_v1_exploitability",
            ]
        )
    command.extend(["--output", str(args.combined_output)])
    return command


def validate_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "validate_phase3_full.py",
        "--combined",
        str(args.combined_output),
        "--report",
        str(args.validation_report),
    ]


def run_batch(args: argparse.Namespace) -> int:
    args.created_at_utc = utc_now()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    specs = phase3_jobs(args.python)
    states = {spec.name: JobState(spec=spec, log_path=args.log_dir / f"{spec.name}.log") for spec in specs}
    env = managed_env(args.torch_threads)

    for state in states.values():
        if args.resume and output_complete(state.spec):
            state.state = "skipped"
            state.started_at_utc = args.created_at_utc
            state.finished_at_utc = args.created_at_utc
            state.elapsed_seconds = 0.0

    write_status(args.status_path, status_payload(args, states, "running"))

    while not all_done(states):
        update_running_jobs(states)
        capacity = max(0, args.max_parallel - len(running_jobs(states)))
        for job in pending_jobs(states)[:capacity]:
            start_job(job, env)
        write_status(args.status_path, status_payload(args, states, "running"))
        if all_done(states):
            break
        time.sleep(args.poll_seconds)

    if any_failed(states):
        write_status(args.status_path, status_payload(args, states, "failed"))
        return 1

    combined_rc = run_command(build_combined_command(args), args.log_dir / "build_combined_table.log", env)
    if combined_rc != 0:
        write_status(args.status_path, status_payload(args, states, "postprocess_failed"))
        return combined_rc

    validation_rc = run_command(validate_command(args), args.log_dir / "validate_phase3_full.log", env)
    overall_state = "complete" if validation_rc == 0 else "validation_failed"
    write_status(args.status_path, status_payload(args, states, overall_state))
    return validation_rc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Phase 3 PyTorch/MARL experiment batch.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase3_full"))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/phase3_full/logs"))
    parser.add_argument("--status-path", type=Path, default=Path("outputs/phase3_full/status.json"))
    parser.add_argument("--combined-output", type=Path, default=Path("outputs/phase3_full/mind_comparison.csv"))
    parser.add_argument("--validation-report", type=Path, default=Path("outputs/phase3_full/validation_report.json"))
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(resume=True)
    args = parser.parse_args(argv)
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be positive")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        for spec in phase3_jobs(args.python):
            print(" ".join(spec.command))
        print(" ".join(build_combined_command(args)))
        print(" ".join(validate_command(args)))
        return
    raise SystemExit(run_batch(args))


if __name__ == "__main__":
    main()
