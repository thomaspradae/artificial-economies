from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_with_pyyaml(path: Path) -> dict[str, Any] | None:
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def _load_simple_queries_yaml(path: Path) -> dict[str, Any]:
    """Parse the restricted YAML shape used by literature/queries.yaml.

    This is intentionally small so the pipeline stays runnable in the repo's
    baseline environment without adding PyYAML as a hard dependency.
    """
    root: dict[str, Any] = {"worlds": {}}
    current_world: str | None = None
    current_group: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0 and stripped == "worlds:":
            continue
        if indent == 2 and stripped.endswith(":"):
            current_world = stripped[:-1]
            root["worlds"][current_world] = {}
            current_group = None
            continue
        if indent == 4 and stripped.endswith(":"):
            if current_world is None:
                raise ValueError(f"group outside world in {path}: {line}")
            current_group = stripped[:-1]
            root["worlds"][current_world][current_group] = []
            continue
        if indent == 6 and stripped.startswith("- "):
            if current_world is None or current_group is None:
                raise ValueError(f"item outside group in {path}: {line}")
            value = stripped[2:].strip()
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            root["worlds"][current_world][current_group].append(value)
            continue
        raise ValueError(f"unsupported YAML line in {path}: {line}")
    return root


def load_queries(path: Path) -> dict[str, Any]:
    return _load_with_pyyaml(path) or _load_simple_queries_yaml(path)


def iter_world_queries(config: dict[str, Any]):
    for world, groups in config.get("worlds", {}).items():
        for group_name in ("classical_terms", "learning_terms"):
            for query in groups.get(group_name, []) or []:
                yield world, group_name, query
