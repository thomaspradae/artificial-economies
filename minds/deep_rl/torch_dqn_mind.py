from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.agent import Agent
from core.registry import register_mind
from minds.deep_rl.features import encode_observation
from minds.deep_rl.numpy_nn import ReplayBuffer, Transition
from minds.deep_rl.torch_optim import DeepMindRMSprop


class StructuredQNet(nn.Module):
    """MLP head for structured observations, replacing the Atari conv stack."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_mind("dqn")
class TorchDQNMind(Agent):
    """PyTorch DQN mind for structured discrete-action worlds.

    The DQN update rule is the audited Atari-style core: replay buffer,
    epsilon-greedy behavior policy, bootstrapped target network, DeepMind
    centered RMSProp, and Huber TD loss. Only the feature encoder and MLP
    observation head are specific to these economic worlds.
    """

    def __init__(
        self,
        action_dim: int = 19,
        obs_dim: int | None = None,
        hidden_dim: int = 64,
        hidden_dims: tuple[int, ...] | None = None,
        lr: float = 2.5e-4,
        gamma: float = 0.96,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.9995,
        replay_size: int = 10_000,
        batch_size: int = 32,
        min_replay_size: int = 32,
        target_update_interval: int = 100,
        grad_clip: float = 10.0,
        seed: int = 0,
        rng: np.random.Generator | None = None,
        device: str = "cpu",
    ):
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else (hidden_dim, hidden_dim)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.target_update_interval = target_update_interval
        self.grad_clip = grad_clip
        self.seed = seed
        self.device = torch.device(device)
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.replay = ReplayBuffer(replay_size, self.rng)
        self.q_net: StructuredQNet | None = None
        self.target_net: StructuredQNet | None = None
        self.optimizer: DeepMindRMSprop | None = None
        self.update_count = 0
        self.last_loss = float("nan")

    def _features(self, obs: Any) -> np.ndarray:
        return encode_observation(obs, obs_dim=self.obs_dim, action_dim=self.action_dim).astype(np.float32)

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _ensure_network(self, obs: Any) -> None:
        if self.q_net is not None:
            return
        features = self._features(obs)
        self.obs_dim = int(len(features))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(self.seed))
            self.q_net = StructuredQNet(self.obs_dim, self.action_dim, self.hidden_dims).to(self.device)
            self.target_net = StructuredQNet(self.obs_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = DeepMindRMSprop(self.q_net.parameters(), lr=self.lr, alpha=0.95, eps=0.01)

    def act(self, obs: Any, epsilon: float | None = None) -> int:
        self._ensure_network(obs)
        exploration = self.epsilon if epsilon is None else epsilon
        if self.rng.random() < exploration:
            return int(self.rng.integers(self.action_dim))
        return self.greedy_action(obs)

    def greedy_action(self, obs: Any) -> int:
        self._ensure_network(obs)
        assert self.q_net is not None
        with torch.no_grad():
            q_values = self.q_net(self._tensor(self._features(obs))[None, :])
        return int(torch.argmax(q_values, dim=1).item())

    def q_values(self, obs: Any) -> np.ndarray:
        """Return Q-values for diagnostics and independence tests."""
        self._ensure_network(obs)
        assert self.q_net is not None
        with torch.no_grad():
            values = self.q_net(self._tensor(self._features(obs))[None, :])
        return values.detach().cpu().numpy()[0]

    def update(self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool) -> None:
        self._ensure_network(obs)
        self._ensure_network(next_obs)
        assert self.q_net is not None and self.target_net is not None and self.optimizer is not None

        self.replay.append(
            Transition(
                obs=self._features(obs),
                action=int(action),
                reward=float(reward),
                next_obs=self._features(next_obs),
                done=bool(done),
            )
        )
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if len(self.replay) < max(self.min_replay_size, self.batch_size):
            return

        batch = self.replay.sample(self.batch_size)
        states = self._tensor(np.vstack([item.obs for item in batch]))
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.long, device=self.device)
        rewards = self._tensor(np.array([item.reward for item in batch], dtype=np.float32))
        next_states = self._tensor(np.vstack([item.next_obs for item in batch]))
        dones = self._tensor(np.array([item.done for item in batch], dtype=np.float32))

        predictions = self.q_net(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = F.smooth_l1_loss(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.last_loss = float(loss.detach().cpu().item())
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def reset(self) -> None:
        """No episodic hidden state; learned parameters are intentionally kept."""
