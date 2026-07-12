"""Multi-agent RL mind wrappers."""

from minds.marl.centralized_critic import CentralizedCriticLearners, CentralizedCriticMind
from minds.marl.independent_learners import IndependentDQNLearners, IndependentDQNMind, IndependentLearners

__all__ = [
    "CentralizedCriticLearners",
    "CentralizedCriticMind",
    "IndependentDQNLearners",
    "IndependentDQNMind",
    "IndependentLearners",
]
