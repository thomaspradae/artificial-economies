from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def clipped_gradient(values: np.ndarray, clip: float) -> np.ndarray:
    if clip <= 0:
        return values
    return np.clip(values, -clip, clip)


class MLPQNetwork:
    """Small one-hidden-layer Q-network with manual NumPy backprop."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int, rng: np.random.Generator):
        if input_dim < 1 or action_dim < 1 or hidden_dim < 1:
            raise ValueError("network dimensions must be positive")
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.w1 = rng.normal(0.0, scale1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.w2 = rng.normal(0.0, scale2, size=(hidden_dim, action_dim))
        self.b2 = np.zeros(action_dim, dtype=float)

    def copy(self) -> "MLPQNetwork":
        clone = object.__new__(MLPQNetwork)
        clone.input_dim = self.input_dim
        clone.action_dim = self.action_dim
        clone.hidden_dim = self.hidden_dim
        clone.w1 = self.w1.copy()
        clone.b1 = self.b1.copy()
        clone.w2 = self.w2.copy()
        clone.b2 = self.b2.copy()
        return clone

    def load_from(self, other: "MLPQNetwork") -> None:
        self.w1[...] = other.w1
        self.b1[...] = other.b1
        self.w2[...] = other.w2
        self.b2[...] = other.b2

    def forward(self, states: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        x = np.atleast_2d(states).astype(float)
        z1 = x @ self.w1 + self.b1
        h = np.maximum(z1, 0.0)
        return h @ self.w2 + self.b2, (x, z1)

    def predict(self, state: np.ndarray) -> np.ndarray:
        q_values, _ = self.forward(np.asarray(state, dtype=float)[None, :])
        return q_values[0]

    def train_q(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        lr: float,
        grad_clip: float,
    ) -> float:
        q_values, (x, z1) = self.forward(states)
        batch = len(x)
        chosen = q_values[np.arange(batch), actions]
        errors = chosen - targets
        loss = float(np.mean(errors**2))

        grad_q = np.zeros_like(q_values)
        grad_q[np.arange(batch), actions] = 2.0 * errors / batch
        hidden = np.maximum(z1, 0.0)

        grad_w2 = hidden.T @ grad_q
        grad_b2 = np.sum(grad_q, axis=0)
        grad_hidden = grad_q @ self.w2.T
        grad_z1 = grad_hidden * (z1 > 0.0)
        grad_w1 = x.T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0)

        self.w2 -= lr * clipped_gradient(grad_w2, grad_clip)
        self.b2 -= lr * clipped_gradient(grad_b2, grad_clip)
        self.w1 -= lr * clipped_gradient(grad_w1, grad_clip)
        self.b1 -= lr * clipped_gradient(grad_b1, grad_clip)
        return loss


class ActorCriticNetwork:
    """One-hidden-layer discrete actor-critic network with manual gradients."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int, rng: np.random.Generator):
        if input_dim < 1 or action_dim < 1 or hidden_dim < 1:
            raise ValueError("network dimensions must be positive")
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.w1 = rng.normal(0.0, scale1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.wp = rng.normal(0.0, scale2, size=(hidden_dim, action_dim))
        self.bp = np.zeros(action_dim, dtype=float)
        self.wv = rng.normal(0.0, scale2, size=(hidden_dim, 1))
        self.bv = np.zeros(1, dtype=float)

    def forward(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        x = np.atleast_2d(states).astype(float)
        z1 = x @ self.w1 + self.b1
        h = np.maximum(z1, 0.0)
        logits = h @ self.wp + self.bp
        values = (h @ self.wv + self.bv).reshape(-1)
        return logits, values, (x, z1, h)

    def policy_value(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        logits, values, _ = self.forward(np.asarray(state, dtype=float)[None, :])
        return softmax(logits)[0], float(values[0])

    def train_actor_critic(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        old_action_probs: np.ndarray,
        clip_ratio: float,
        policy_lr: float,
        value_lr: float,
        entropy_coef: float,
        grad_clip: float,
    ) -> dict[str, float]:
        logits, values, (x, z1, h) = self.forward(states)
        probs = softmax(logits)
        batch = len(x)
        action_probs = np.clip(probs[np.arange(batch), actions], 1e-8, 1.0)
        old = np.clip(old_action_probs, 1e-8, 1.0)
        ratios = action_probs / old

        active = np.ones(batch, dtype=float)
        active[(advantages >= 0.0) & (ratios > 1.0 + clip_ratio)] = 0.0
        active[(advantages < 0.0) & (ratios < 1.0 - clip_ratio)] = 0.0

        one_hot_actions = np.zeros_like(probs)
        one_hot_actions[np.arange(batch), actions] = 1.0
        grad_logits = -(active * advantages * ratios)[:, None] * (one_hot_actions - probs) / batch

        if entropy_coef:
            entropy_grad = probs * (np.log(np.clip(probs, 1e-8, 1.0)) + 1.0)
            grad_logits += entropy_coef * entropy_grad / batch

        value_errors = values - returns
        grad_values = 2.0 * value_errors / batch

        grad_wp = h.T @ grad_logits
        grad_bp = np.sum(grad_logits, axis=0)
        grad_wv = h.T @ grad_values[:, None]
        grad_bv = np.array([np.sum(grad_values)])

        grad_h = grad_logits @ self.wp.T + grad_values[:, None] @ self.wv.T
        grad_z1 = grad_h * (z1 > 0.0)
        grad_w1 = x.T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0)

        self.wp -= policy_lr * clipped_gradient(grad_wp, grad_clip)
        self.bp -= policy_lr * clipped_gradient(grad_bp, grad_clip)
        self.wv -= value_lr * clipped_gradient(grad_wv, grad_clip)
        self.bv -= value_lr * clipped_gradient(grad_bv, grad_clip)
        self.w1 -= policy_lr * clipped_gradient(grad_w1, grad_clip)
        self.b1 -= policy_lr * clipped_gradient(grad_b1, grad_clip)

        unclipped = ratios * advantages
        clipped = np.clip(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        return {
            "policy_objective": float(np.mean(np.minimum(unclipped, clipped))),
            "value_loss": float(np.mean(value_errors**2)),
            "entropy": float(-np.mean(np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1))),
        }


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size transition replay buffer."""

    def __init__(self, capacity: int, rng: np.random.Generator):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.rng = rng
        self.items: Deque[Transition] = deque(maxlen=capacity)

    def append(self, transition: Transition) -> None:
        self.items.append(transition)

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self.items):
            raise ValueError("batch_size exceeds replay size")
        indices = self.rng.choice(len(self.items), size=batch_size, replace=False)
        values = list(self.items)
        return [values[int(index)] for index in indices]
