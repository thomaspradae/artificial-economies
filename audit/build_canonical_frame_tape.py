#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from common import read_jsonl


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs")
    parser = argparse.ArgumentParser(description="Write a freeze manifest for a canonical DeepMind frame tape.")
    parser.add_argument("--tape-dir", default=os.path.join(out, "canonical_frames"))
    parser.add_argument("--out", default=os.path.join(out, "freeze_manifest.json"))
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--dm-dir", default=os.getenv("DM_DIR"))
    return parser.parse_args()


def git_commit(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def file_sha256(path: str | Path) -> str | None:
    try:
        import hashlib

        h = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    tape_dir = Path(args.tape_dir)
    manifest_path = tape_dir / "manifest.json"
    transitions_path = tape_dir / "transitions.jsonl"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    transitions = [row for row in read_jsonl(transitions_path) if row.get("phase") == "transition"]
    first = transitions[0] if transitions else {}
    last = transitions[-1] if transitions else {}

    freeze_manifest = {
        "phase": "freeze_manifest",
        "canonical_source": manifest.get("canonical_source", "deepmind_alewrap"),
        "game": manifest.get("game"),
        "seed": manifest.get("seed"),
        "frame_skip": manifest.get("frame_skip"),
        "transition_count": len(transitions),
        "action_tape_txt": manifest.get("action_tape_txt"),
        "rom": manifest.get("rom"),
        "rom_sha256": file_sha256(manifest.get("rom")),
        "tape_dir": str(tape_dir),
        "transitions_path": str(transitions_path),
        "first_pooled_hash": (first.get("pooled_frame") or {}).get("hash"),
        "last_pooled_hash": (last.get("pooled_frame") or {}).get("hash"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repo_root": str(Path(args.repo_root).resolve()),
        "repo_commit": git_commit(args.repo_root),
        "deepmind_dir": args.dm_dir,
        "deepmind_commit": git_commit(args.dm_dir),
        "script_versions": {
            "canonical_builder": "audit/deepmind/build_canonical_env_tape.lua",
            "pytorch_preprocess": "audit/pytorch/trace_preprocess.py",
            "deepmind_preprocess": "audit/deepmind/trace_preprocess.lua",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(freeze_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
