#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from common import read_jsonl


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs")
    parser = argparse.ArgumentParser(description="Diagnose Stage 1 action/repeat/pooling mismatch.")
    parser.add_argument("--pytorch", default=os.path.join(out, "pytorch_env.jsonl"))
    parser.add_argument("--deepmind", default=os.path.join(out, "deepmind_env.jsonl"))
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--out", default=os.getenv("ACTION_REPEAT_DIAGNOSIS_OUT"))
    return parser.parse_args()


def find_step(rows: list[dict[str, Any]], step: int) -> dict[str, Any]:
    for row in rows:
        if row.get("phase") == "agent_step" and int(row.get("step", -1)) == step:
            return row
    raise SystemExit(f"missing agent_step {step}")


def val(row: dict[str, Any], key: str) -> Any:
    return row.get(key)


def frame_hash(repeat: dict[str, Any]) -> Any:
    raw = repeat.get("raw_frame")
    return raw.get("hash") if isinstance(raw, dict) else None


def yes_no(value: bool) -> str:
    return "true" if value else "false"


def main() -> int:
    args = parse_args()
    pytorch = find_step(read_jsonl(args.pytorch), args.step)
    deepmind = find_step(read_jsonl(args.deepmind), args.step)

    p_repeats = pytorch.get("repeats", [])
    d_repeats = deepmind.get("repeats", [])
    repeat_pairs = list(zip(p_repeats, d_repeats))

    same_action_code = val(pytorch, "ale_action_code_used") == val(deepmind, "ale_action_code_used")
    same_action_index = val(pytorch, "action_index_used") == val(deepmind, "action_index_used")
    same_repeat_count = val(pytorch, "repeat_count") == val(deepmind, "repeat_count")
    same_pooling_indices = val(pytorch, "pooling_frame_indices") == val(deepmind, "pooling_frame_indices")
    same_pooled_hash = (
        pytorch.get("pooled_frame", {}).get("hash") == deepmind.get("pooled_frame", {}).get("hash")
    )

    first_repeat_mismatch = "none"
    for index, (p_rep, d_rep) in enumerate(repeat_pairs):
        if frame_hash(p_rep) != frame_hash(d_rep):
            first_repeat_mismatch = str(index)
            break

    lines = [
        f"Action/repeat diagnosis for agent_step {args.step}",
        "",
        f"pytorch_tape_action_raw: {val(pytorch, 'tape_action_raw')}",
        f"deepmind_tape_action_raw: {val(deepmind, 'tape_action_raw')}",
        f"pytorch_action_index_used: {val(pytorch, 'action_index_used')}",
        f"deepmind_action_index_used: {val(deepmind, 'action_index_used')}",
        f"pytorch_ale_action_code_used: {val(pytorch, 'ale_action_code_used')}",
        f"deepmind_ale_action_code_used: {val(deepmind, 'ale_action_code_used')}",
        f"pytorch_action_meaning_used: {val(pytorch, 'action_meaning_used')}",
        f"deepmind_action_meaning_used: {val(deepmind, 'action_meaning_used')}",
        f"same_action_index: {yes_no(same_action_index)}",
        f"same_ale_action_code: {yes_no(same_action_code)}",
        f"same_repeat_count: {yes_no(same_repeat_count)}",
        f"same_pooling_frame_indices: {yes_no(same_pooling_indices)}",
        f"same_pooled_frame_hash: {yes_no(same_pooled_hash)}",
        f"first_per_repeat_frame_mismatch: {first_repeat_mismatch}",
        "",
        "Per-repeat comparison",
    ]
    for index, (p_rep, d_rep) in enumerate(repeat_pairs):
        lines.extend(
            [
                f"repeat {index}:",
                f"  pytorch_hash: {frame_hash(p_rep)}",
                f"  deepmind_hash: {frame_hash(d_rep)}",
                f"  frame_hash_equal: {yes_no(frame_hash(p_rep) == frame_hash(d_rep))}",
                f"  pytorch_reward: {p_rep.get('reward')}",
                f"  deepmind_reward: {d_rep.get('reward')}",
                f"  pytorch_lives_after: {p_rep.get('lives_after')}",
                f"  deepmind_lives_after: {d_rep.get('lives_after')}",
                f"  pytorch_terminal: {p_rep.get('terminal')}",
                f"  deepmind_terminal: {d_rep.get('terminal')}",
            ]
        )

    lines.extend(["", "Conclusion:"])
    if not same_action_code:
        lines.append("action code differs; action tape/mapping bug found")
    elif not same_repeat_count:
        lines.append("action code matches but repeat count differs")
    elif first_repeat_mismatch != "none":
        lines.append("same action code and repeat count, but per-repeat frames diverge; suspect ALE step boundary/backend state")
    elif not same_pooling_indices or not same_pooled_hash:
        lines.append("per-repeat frames match but pooled output differs; pooling implementation mismatch")
    else:
        lines.append("agent_step action, repeats, and pooled frame match")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
