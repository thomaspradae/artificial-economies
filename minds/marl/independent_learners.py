from __future__ import annotations

from typing import Any

import numpy as np

from core.registry import register_mind
from minds.deep_rl.dqn_mind import DQNMind
from minds.deep_rl.ppo_mind import PPOMind
from minds.deep_rl.torch_dqn_mind import TorchDQNMind


def derive_independent_seed(base_seed: int, stream_index: int) -> int:
    """Derive statistically separated child seeds from one experiment seed."""
    if stream_index < 0:
        raise ValueError("stream_index must be non-negative")
    child = np.random.SeedSequence(int(base_seed)).spawn(stream_index + 1)[stream_index]
    return int(child.generate_state(1, dtype=np.uint32)[0])


@register_mind("independent_dqn")
class IndependentDQNMind(DQNMind):
    """Standalone decorrelated DQN mind for direct registry construction.

    Full multi-agent runs use `IndependentDQNLearners` below so every agent
    receives a child stream from one base experiment seed. This class keeps
    the registry entry useful while avoiding bit-identical behavior to `dqn`
    when constructed directly.
    """

    def __init__(self, *args: Any, seed: int = 0, stream_index: int = 0, **kwargs: Any):
        derived_seed = derive_independent_seed(seed, stream_index)
        super().__init__(*args, seed=derived_seed, rng=np.random.default_rng(derived_seed), **kwargs)


class IndependentDQNLearners:
    """Genuine independent-learners DQN baseline.

    Each agent owns a separately initialized DQN, replay buffer, and
    epsilon-greedy RNG stream. The learners share only the environment.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        base_seed: int,
        **dqn_params: Any,
    ):
        if n_agents < 1:
            raise ValueError("n_agents must be positive")
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.base_seed = int(base_seed)
        self.agents = []
        for index in range(n_agents):
            seed = derive_independent_seed(self.base_seed, index)
            self.agents.append(
                TorchDQNMind(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    seed=seed,
                    rng=np.random.default_rng(seed),
                    **dqn_params,
                )
            )

    def act(self, observations: list[Any]) -> list[int]:
        if len(observations) != len(self.agents):
            raise ValueError("one observation per independent DQN agent is required")
        return [int(agent.act(obs)) for agent, obs in zip(self.agents, observations)]

    def update(
        self,
        observations: list[Any],
        actions: list[int],
        rewards: list[float],
        next_observations: list[Any],
        dones: list[bool],
    ) -> None:
        for agent, obs, action, reward, next_obs, done in zip(
            self.agents,
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ):
            agent.update(obs, action, reward, next_obs, done)


class IndependentLearners:
    """Coordinator for N independent DQN/PPO learners with no shared state."""

    def __init__(self, n_agents: int, mind: str = "dqn", **mind_params: Any):
        if n_agents < 1:
            raise ValueError("n_agents must be positive")
        cls = DQNMind if mind == "dqn" else PPOMind if mind == "ppo" else None
        if cls is None:
            raise ValueError("mind must be 'dqn' or 'ppo'")
        seed = int(mind_params.pop("seed", 0))
        self.agents = []
        for index in range(n_agents):
            derived_seed = derive_independent_seed(seed, index)
            params = dict(mind_params)
            if cls is DQNMind:
                params["rng"] = np.random.default_rng(derived_seed)
            self.agents.append(cls(seed=derived_seed, **params))

    def act(self, observations: list[Any]) -> list[int]:
        if len(observations) != len(self.agents):
            raise ValueError("one observation per agent is required")
        return [int(agent.act(obs)) for agent, obs in zip(self.agents, observations)]

    def update(
        self,
        observations: list[Any],
        actions: list[int],
        rewards: list[float],
        next_observations: list[Any],
        dones: list[bool],
    ) -> None:
        for agent, obs, action, reward, next_obs, done in zip(
            self.agents,
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ):
            agent.update(obs, action, reward, next_obs, done)
