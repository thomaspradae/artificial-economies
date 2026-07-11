from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from core.agent import Agent
from core.registry import register_mind
from minds.deep_rl.features import encode_observation


class StructuredPolicyNet(nn.Module):
    """Categorical policy MLP for structured observations."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.Tanh()])
            prev = width
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StructuredValueNet(nn.Module):
    """State-value MLP for structured observations."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.Tanh()])
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TorchRolloutItem:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    old_log_prob: float
    value: float
    next_value: float


@register_mind("ppo")
class TorchPPOMind(Agent):
    """PyTorch PPO mind with discrete Categorical actions.

    This keeps the PPO core intact: GAE, clipped surrogate objective, value
    loss, entropy bonus, and repeated minibatch updates. The architecture
    change is the structured MLP policy/value heads and categorical action
    distribution for discrete economic environments.
    """

    def __init__(
        self,
        action_dim: int = 19,
        obs_dim: int | None = None,
        hidden_dim: int = 64,
        hidden_dims: tuple[int, ...] | None = None,
        gamma: float = 0.96,
        gae_lambda: float = 0.95,
        rollout_steps: int = 32,
        epochs: int = 3,
        batch_size: int = 32,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        grad_clip: float = 5.0,
        seed: int = 0,
        device: str = "cpu",
    ):
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else (hidden_dim, hidden_dim)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout_steps = rollout_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.grad_clip = grad_clip
        self.seed = seed
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.policy_net: StructuredPolicyNet | None = None
        self.value_net: StructuredValueNet | None = None
        self.policy_optimizer: torch.optim.Optimizer | None = None
        self.value_optimizer: torch.optim.Optimizer | None = None
        self.rollout: list[TorchRolloutItem] = []
        self._last_action: int | None = None
        self._last_log_prob: float | None = None
        self._last_value: float | None = None
        self.last_update: dict[str, float] = {}

    def _features(self, obs: Any) -> np.ndarray:
        return encode_observation(obs, obs_dim=self.obs_dim, action_dim=self.action_dim).astype(np.float32)

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _ensure_network(self, obs: Any) -> None:
        if self.policy_net is not None:
            return
        features = self._features(obs)
        self.obs_dim = int(len(features))
        self.policy_net = StructuredPolicyNet(self.obs_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.value_net = StructuredValueNet(self.obs_dim, self.hidden_dims).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr)

    def _policy_value(self, obs: Any) -> tuple[Categorical, torch.Tensor]:
        self._ensure_network(obs)
        assert self.policy_net is not None and self.value_net is not None
        features = self._tensor(self._features(obs))[None, :]
        logits = self.policy_net(features)
        value = self.value_net(features)[0]
        return Categorical(logits=logits), value

    def act(self, obs: Any) -> int:
        dist, value = self._policy_value(obs)
        action_tensor = dist.sample()
        action = int(action_tensor.item())
        self._last_action = action
        self._last_log_prob = float(dist.log_prob(action_tensor).detach().cpu().item())
        self._last_value = float(value.detach().cpu().item())
        return action

    def greedy_action(self, obs: Any) -> int:
        self._ensure_network(obs)
        assert self.policy_net is not None
        with torch.no_grad():
            logits = self.policy_net(self._tensor(self._features(obs))[None, :])
        return int(torch.argmax(logits, dim=1).item())

    def update(self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool) -> None:
        dist, value = self._policy_value(obs)
        _, next_value = self._policy_value(next_obs)
        action_tensor = torch.as_tensor(int(action), dtype=torch.long, device=self.device)
        old_log_prob = float(dist.log_prob(action_tensor).detach().cpu().item())
        old_value = float(value.detach().cpu().item())
        if self._last_action == int(action) and self._last_log_prob is not None and self._last_value is not None:
            old_log_prob = self._last_log_prob
            old_value = self._last_value

        self.rollout.append(
            TorchRolloutItem(
                obs=self._features(obs),
                action=int(action),
                reward=float(reward),
                next_obs=self._features(next_obs),
                done=bool(done),
                old_log_prob=old_log_prob,
                value=old_value,
                next_value=float(next_value.detach().cpu().item()),
            )
        )
        if len(self.rollout) >= self.rollout_steps or done:
            self._train_rollout()

    def update_with_advantage(self, obs: Any, action: Any, advantage: float, value_target: float | None = None) -> None:
        self._ensure_network(obs)
        assert self.policy_net is not None and self.value_net is not None
        assert self.policy_optimizer is not None and self.value_optimizer is not None
        states = self._tensor(self._features(obs))[None, :]
        action_tensor = torch.as_tensor([int(action)], dtype=torch.long, device=self.device)
        advantage_tensor = torch.as_tensor([float(advantage)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            current_value = self.value_net(states)
            target = current_value + advantage_tensor if value_target is None else torch.as_tensor([float(value_target)], dtype=torch.float32, device=self.device)

        logits = self.policy_net(states)
        dist = Categorical(logits=logits)
        policy_loss = -(dist.log_prob(action_tensor) * advantage_tensor).mean() - self.entropy_coef * dist.entropy().mean()
        value_loss = F.mse_loss(self.value_net(states), target)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip)
        self.value_optimizer.step()

        self.last_update = {
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "entropy": float(dist.entropy().mean().detach().cpu().item()),
        }

    def _train_rollout(self) -> None:
        if not self.rollout:
            return
        self._ensure_network(self.rollout[0].obs)
        assert self.policy_net is not None and self.value_net is not None
        assert self.policy_optimizer is not None and self.value_optimizer is not None

        rewards = np.array([item.reward for item in self.rollout], dtype=np.float32)
        values = np.array([item.value for item in self.rollout], dtype=np.float32)
        next_values = np.array([item.next_value for item in self.rollout], dtype=np.float32)
        dones = np.array([item.done for item in self.rollout], dtype=np.float32)
        deltas = rewards + (1.0 - dones) * self.gamma * next_values - values

        advantages = np.zeros_like(deltas)
        gae = 0.0
        for index in range(len(deltas) - 1, -1, -1):
            gae = float(deltas[index]) + (1.0 - float(dones[index])) * self.gamma * self.gae_lambda * gae
            advantages[index] = gae
        returns = advantages + values
        if len(advantages) > 1 and float(np.std(advantages)) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        states = self._tensor(np.vstack([item.obs for item in self.rollout]))
        actions = torch.as_tensor([item.action for item in self.rollout], dtype=torch.long, device=self.device)
        old_log_probs = self._tensor(np.array([item.old_log_prob for item in self.rollout], dtype=np.float32))
        advantages_t = self._tensor(advantages)
        returns_t = self._tensor(returns)

        n = len(self.rollout)
        effective_batch = min(self.batch_size, n)
        for _ in range(self.epochs):
            indices = self.rng.permutation(n)
            for start in range(0, n, effective_batch):
                batch_np = indices[start : start + effective_batch]
                batch = torch.as_tensor(batch_np, dtype=torch.long, device=self.device)
                batch_states = states.index_select(0, batch)
                batch_actions = actions.index_select(0, batch)
                batch_old_log_probs = old_log_probs.index_select(0, batch)
                batch_advantages = advantages_t.index_select(0, batch)
                batch_returns = returns_t.index_select(0, batch)

                logits = self.policy_net(batch_states)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions)
                ratios = torch.exp(log_probs - batch_old_log_probs)
                unclipped = ratios * batch_advantages
                clipped = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
                entropy = dist.entropy().mean()

                values = self.value_net(batch_states)
                value_loss = F.mse_loss(values, batch_returns)

                self.policy_optimizer.zero_grad()
                (policy_loss - self.entropy_coef * entropy).backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip)
                self.value_optimizer.step()

                self.last_update = {
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "entropy": float(entropy.detach().cpu().item()),
                }

        self.rollout = []

    def reset(self) -> None:
        self._last_action = None
        self._last_log_prob = None
        self._last_value = None
