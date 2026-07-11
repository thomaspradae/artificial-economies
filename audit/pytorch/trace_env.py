#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import canonical_float, frame_stats, frame_to_uint8_hwc, ram_stats, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace Gymnasium/ALE env facts.")
    parser.add_argument("--env-id", default=os.getenv("ENV_ID", "ALE/Breakout-v5"))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "1")))
    parser.add_argument(
        "--steps", type=int, default=int(os.getenv("TRACE_STEPS", "200"))
    )
    parser.add_argument(
        "--frame-skip", type=int, default=int(os.getenv("FRAME_SKIP", "4"))
    )
    parser.add_argument("--action-tape", required=True)
    parser.add_argument("--out", default=os.getenv("PYTORCH_ENV_OUT", "pytorch_env.jsonl"))
    parser.add_argument(
        "--frame-dump-dir",
        default=os.getenv("FRAME_DUMP_DIR"),
        help="Directory for init/step frame .npy and optional .png dumps.",
    )
    parser.add_argument(
        "--frame-dump-steps",
        type=int,
        default=int(os.getenv("FRAME_DUMP_STEPS", "20")),
        help="Dump agent-step frames 0 through this index, inclusive.",
    )
    parser.add_argument(
        "--full-action-space",
        action="store_true",
        help="Request Gymnasium full_action_space=True.",
    )
    parser.add_argument(
        "--action-tape-mode",
        choices=("index", "ale_code"),
        default=os.getenv("ACTION_TAPE_MODE", "index"),
    )
    return parser.parse_args()


def load_action_tape(path: str | Path) -> np.ndarray:
    tape_path = Path(path)
    if tape_path.suffix == ".npy":
        actions = np.load(tape_path)
    else:
        actions = np.loadtxt(tape_path, dtype=np.int64)
    return np.asarray(actions, dtype=np.int64).reshape(-1)


def make_env(env_id: str, full_action_space: bool) -> Any:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit(
            "gymnasium is required for audit/pytorch/trace_env.py"
        ) from exc
    try:
        import ale_py

        gym.register_envs(ale_py)
    except Exception:
        # Older Gymnasium/ALE installs may register Atari environments at
        # import time. If not, gym.make will raise a clear error below.
        pass

    attempts = [
        {
            "obs_type": "rgb",
            "frameskip": 1,
            "repeat_action_probability": 0.0,
            "full_action_space": full_action_space,
        },
        {
            "frameskip": 1,
            "repeat_action_probability": 0.0,
            "full_action_space": full_action_space,
        },
        {"frameskip": 1, "repeat_action_probability": 0.0},
        {},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return gym.make(env_id, **kwargs)
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"could not create env {env_id}")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def unwrap_reset(reset_result: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result
    return reset_result, {}


def safe_ale_attr(env: Any, method_name: str) -> Any:
    ale = getattr(getattr(env, "unwrapped", env), "ale", None)
    if ale is None or not hasattr(ale, method_name):
        return None
    try:
        return getattr(ale, method_name)()
    except Exception:
        return None


def safe_ale_ram(env: Any) -> dict[str, Any] | None:
    return ram_stats(safe_ale_attr(env, "getRAM"))


def safe_score(env: Any, info: dict[str, Any] | None = None) -> Any:
    if info:
        for key in ("score", "episode_score", "ale.score"):
            if key in info:
                return info[key]
    for method_name in ("getEpisodeScore", "getScore", "score"):
        value = safe_ale_attr(env, method_name)
        if value is not None:
            return value
    return None


def action_meanings(env: Any) -> list[str] | None:
    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "get_action_meanings"):
        return list(unwrapped.get_action_meanings())
    if hasattr(env, "get_action_meanings"):
        return list(env.get_action_meanings())
    return None


def action_values(env: Any) -> list[int] | None:
    ale = getattr(getattr(env, "unwrapped", env), "ale", None)
    if ale is not None:
        for method_name in ("getMinimalActionSet", "getLegalActionSet"):
            if hasattr(ale, method_name):
                try:
                    return [int(item) for item in getattr(ale, method_name)()]
                except Exception:
                    pass
    return list(range(int(env.action_space.n)))


def action_meaning_for(env: Any, action_index: int) -> str | None:
    meanings = action_meanings(env)
    if meanings is None or action_index < 0 or action_index >= len(meanings):
        return None
    return meanings[action_index]


def map_tape_action(env: Any, tape_action: int, mode: str) -> tuple[int, int | None, str | None]:
    legal_actions = action_values(env)
    if mode == "index":
        action_index = int(tape_action)
        if legal_actions is not None and 0 <= action_index < len(legal_actions):
            ale_action_code = int(legal_actions[action_index])
        else:
            ale_action_code = action_index
        return action_index, ale_action_code, action_meaning_for(env, action_index)
    action_index = int(tape_action)
    return action_index, int(tape_action), action_meaning_for(env, action_index)


def max_pool_last_two(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("cannot pool empty frame list")
    if len(frames) == 1:
        return frames[-1]
    return np.maximum(frames[-1], frames[-2])


def dump_frame(frame: Any, frame_dir: Path | None, name: str) -> None:
    if frame_dir is None:
        return
    frame_dir.mkdir(parents=True, exist_ok=True)
    arr = frame_to_uint8_hwc(frame)
    np.save(frame_dir / f"{name}.npy", arr)
    if os.getenv("FRAME_DUMP_PNG", "1") == "0":
        return
    try:
        from PIL import Image
    except Exception:
        return

    try:
        if arr.ndim == 2:
            image = Image.fromarray(arr)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            image = Image.fromarray(arr[..., 0])
        elif arr.ndim == 3 and arr.shape[-1] >= 3:
            image = Image.fromarray(arr[..., :3])
        else:
            return
        image.save(frame_dir / f"{name}.png")
    except Exception:
        return


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.frame_skip <= 0:
        raise SystemExit("--frame-skip must be positive")

    actions = load_action_tape(args.action_tape)
    env = make_env(args.env_id, args.full_action_space or args.action_tape_mode == "ale_code")
    frame_dir = Path(args.frame_dump_dir) if args.frame_dump_dir else Path(args.out).parent / "frames"
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(args.seed)

    obs, reset_info = unwrap_reset(env.reset(seed=args.seed))
    obs_arr = np.asarray(obs)
    dump_frame(obs_arr, frame_dir, "pytorch_init")

    rows: list[dict[str, Any]] = [
        {
            "phase": "init",
            "source": "pytorch",
            "env_id": args.env_id,
            "seed": args.seed,
            "frame_skip": args.frame_skip,
            "action_tape_mode": args.action_tape_mode,
            "gymnasium_version": package_version("gymnasium"),
            "ale_py_version": package_version("ale-py") or package_version("ale_py"),
            "alewrap_version": None,
            "ale_version": None,
            "action_space_n": int(env.action_space.n),
            "action_values": action_values(env),
            "legal_action_set": action_values(env),
            "minimal_action_set": action_values(env),
            "action_meanings": action_meanings(env),
            "ale_lives": safe_ale_attr(env, "lives"),
            "ale_frame_number": safe_ale_attr(env, "getFrameNumber"),
            "frame_number": safe_ale_attr(env, "getFrameNumber"),
            "score": safe_score(env),
            "ram": safe_ale_ram(env),
            "reset_info": reset_info,
            "raw_frame": frame_stats(obs_arr),
        }
    ]

    done = False
    for step, action_value in enumerate(actions[: args.steps]):
        if done:
            rows.append({"phase": "trace_end", "source": "pytorch", "step": step, "reason": "done"})
            break

        tape_action_raw = int(action_value)
        action, ale_action_code, action_meaning = map_tape_action(env, tape_action_raw, args.action_tape_mode)
        if action < 0 or action >= int(env.action_space.n):
            raise SystemExit(
                f"action {action} from tape value {tape_action_raw} at step {step} is outside action space "
                f"0..{int(env.action_space.n) - 1}"
            )

        lives_before = safe_ale_attr(env, "lives")
        if step <= args.frame_dump_steps:
            dump_frame(obs_arr, frame_dir, f"pytorch_step_{step:03d}_pre")
        frames: list[np.ndarray] = []
        per_frame_rewards: list[Any] = []
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        total_reward = 0.0
        repeats: list[dict[str, Any]] = []

        for repeat_index in range(args.frame_skip):
            obs, reward, terminated, truncated, info = env.step(action)
            obs_arr = np.asarray(obs)
            frames.append(obs_arr)
            if step <= args.frame_dump_steps:
                dump_frame(obs_arr, frame_dir, f"pytorch_step_{step:03d}_repeat_{repeat_index:03d}")
            per_frame_rewards.append(canonical_float(reward))
            total_reward += float(reward)
            repeats.append(
                {
                    "repeat_i": repeat_index,
                    "tape_action_raw": tape_action_raw,
                    "action_index_used": action,
                    "ale_action_code_used": ale_action_code,
                    "action_meaning_used": action_meaning,
                    "reward": canonical_float(reward),
                    "lives_after": safe_ale_attr(env, "lives"),
                    "terminal": bool(terminated),
                    "truncated": bool(truncated),
                    "frame_number": safe_ale_attr(env, "getFrameNumber"),
                    "score": safe_score(env, info),
                    "ram": safe_ale_ram(env),
                    "raw_frame": frame_stats(obs_arr),
                }
            )
            if terminated or truncated:
                break

        done = bool(terminated or truncated)
        pooled = max_pool_last_two(frames)
        pooling_frame_indices = [len(frames) - 2, len(frames) - 1] if len(frames) >= 2 else [len(frames) - 1]
        if step <= args.frame_dump_steps:
            dump_frame(frames[-1], frame_dir, f"pytorch_step_{step:03d}")
            if len(frames) >= 2:
                dump_frame(frames[-2], frame_dir, f"pytorch_step_{step:03d}_pool_src_{len(frames) - 2:03d}")
                dump_frame(frames[-1], frame_dir, f"pytorch_step_{step:03d}_pool_src_{len(frames) - 1:03d}")
            dump_frame(pooled, frame_dir, f"pytorch_step_{step:03d}_pooled")
        rows.append(
            {
                "phase": "agent_step",
                "source": "pytorch",
                "step": step,
                "action": action,
                "tape_action_raw": tape_action_raw,
                "action_index_used": action,
                "ale_action_code_used": ale_action_code,
                "action_meaning_used": action_meaning,
                "legal_action_set": action_values(env),
                "minimal_action_set": action_values(env),
                "lua_action": None,
                "ale_action": ale_action_code,
                "repeat_count": len(frames),
                "repeats": repeats,
                "per_frame_rewards": per_frame_rewards,
                "reward": canonical_float(total_reward),
                "lives_before": lives_before,
                "lives_after": safe_ale_attr(env, "lives"),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "done": done,
                "ale_frame_number": safe_ale_attr(env, "getFrameNumber"),
                "frame_number": safe_ale_attr(env, "getFrameNumber"),
                "score": safe_score(env, info),
                "ram": safe_ale_ram(env),
                "pooling_frame_indices": pooling_frame_indices,
                "raw_frame": frame_stats(frames[-1]),
                "pooled_frame": frame_stats(pooled),
                "info": info,
            }
        )

    env.close()
    write_jsonl(args.out, rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
