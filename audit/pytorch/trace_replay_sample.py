#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import write_jsonl  # noqa: E402
from stage4_common import batch_plan, build_batch, load_records, load_requested_indices, sample_at  # noqa: E402


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage4_replay_sample")
    parser = argparse.ArgumentParser(description="Trace PyTorch replay sampling semantics for fixed Stage 4 indices.")
    parser.add_argument("--replay-dir", default=os.path.join(out, "canonical_replay"))
    parser.add_argument("--requested", default=os.path.join(out, "requested_indices.txt"))
    parser.add_argument("--out", default=os.path.join(out, "pytorch_batch.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.replay_dir)
    requested = load_requested_indices(args.requested)
    rows = []
    for request_position, requested_index in enumerate(requested):
        row = sample_at(records, requested_index)
        row.update(
            {
                "phase": "replay_sample",
                "source": "pytorch",
                "step": request_position,
                "request_position": request_position,
            }
        )
        rows.append(row)

    for batch_name, indices in batch_plan(records, requested):
        row = build_batch(records, indices, batch_name)
        row.update({"source": "pytorch", "step": f"batch:{batch_name}"})
        rows.append(row)

    write_jsonl(args.out, rows)
    print(f"wrote {args.out}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
