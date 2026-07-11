from __future__ import annotations

from typing import Any, Callable, TypeVar


T = TypeVar("T")

WORLDS: dict[str, type[Any]] = {}
MINDS: dict[str, type[Any]] = {}
INSTITUTIONS: dict[str, type[Any]] = {}


def register_world(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        WORLDS[name] = cls
        return cls

    return deco


def register_mind(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        MINDS[name] = cls
        return cls

    return deco


def register_institution(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        INSTITUTIONS[name] = cls
        return cls

    return deco


def build_experiment(config: dict[str, Any]) -> Any:
    """Instantiate a registered world from a minimal config dict."""
    world_name = config["world"]
    mind_name = config.get("mind")
    institution_name = config.get("institution", "none")
    seed = config.get("seed")
    world_params = config.get("world_params", {})
    mind_params = config.get("mind_params", {})
    institution_params = config.get("institution_params", {})
    n_agents = int(config.get("n_agents", 2))

    if world_name not in WORLDS:
        raise KeyError(f"Unknown world {world_name!r}")
    if institution_name not in INSTITUTIONS:
        raise KeyError(f"Unknown institution {institution_name!r}")

    institution = INSTITUTIONS[institution_name](**institution_params)
    agents = []
    agent_configs = config.get("agents")
    if agent_configs is not None:
        for agent_config in agent_configs:
            agent_mind_name = agent_config["mind"]
            if agent_mind_name not in MINDS:
                raise KeyError(f"Unknown mind {agent_mind_name!r}")
            agents.append(MINDS[agent_mind_name](**agent_config.get("params", {})))
    elif mind_name is not None:
        if mind_name not in MINDS:
            raise KeyError(f"Unknown mind {mind_name!r}")
        agents = [MINDS[mind_name](**mind_params) for _ in range(n_agents)]

    return WORLDS[world_name](agents=agents, institution=institution, seed=seed, **world_params)
