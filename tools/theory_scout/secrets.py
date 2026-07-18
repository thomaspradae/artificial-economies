from __future__ import annotations

import os
from pathlib import Path


def _clean_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: Path, overwrite: bool = False) -> dict[str, bool]:
    """Load a small shell-style env file without printing secret values."""
    loaded: dict[str, bool] = {}
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if key in os.environ and not overwrite:
            loaded[key] = True
            continue
        os.environ[key] = _clean_value(value)
        loaded[key] = True
    return loaded


def secret_presence() -> dict[str, bool]:
    keys = [
        "OPENALEX_API_KEY",
        "OPENALEX_MAILTO",
        "SEMANTIC_SCHOLAR_API_KEY",
        "UNPAYWALL_EMAIL",
    ]
    return {key: bool(os.getenv(key)) for key in keys}
