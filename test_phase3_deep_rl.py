import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from minds.deep_rl.dqn_mind import DQNMind
from minds.deep_rl.features import encode_observation
from minds.deep_rl.ppo_mind import PPOMind
from minds.deep_rl.simple_dqn_mind import SimpleDQNMind
from minds.deep_rl.simple_ppo_mind import SimplePPOMind
from minds.deep_rl.torch_dqn_mind import StructuredQNet, TorchDQNMind
from minds.deep_rl.torch_ppo_mind import StructuredPolicyNet, StructuredValueNet, TorchPPOMind
from minds.marl.centralized_critic import CentralizedCriticLearners
from minds.marl.independent_learners import IndependentLearners
from run_phase3_validation import run_validation, parse_args
from worlds.pricing_arena.training import train_market


class DeepRLFeatureTests(unittest.TestCase):
    def test_pricing_tuple_observation_is_two_one_hot_vectors(self):
        vector = encode_observation((1, 3), action_dim=5)
        self.assertEqual(len(vector), 10)
        self.assertEqual(vector[1], 1.0)
        self.assertEqual(vector[5 + 3], 1.0)
        self.assertEqual(np.sum(vector), 2.0)

    def test_torch_forward_backward_smoke(self):
        net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        x = torch.ones(5, 3)
        y = net(x).sum()
        y.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in net.parameters()))


class DeepRLMindTests(unittest.TestCase):
    def test_dqn_mind_outputs_valid_actions_and_updates(self):
        mind = DQNMind(action_dim=5, obs_dim=10, hidden_dim=8, batch_size=2, min_replay_size=2, seed=4)
        self.assertIsInstance(mind, TorchDQNMind)
        obs = (1, 2)
        next_obs = (2, 3)
        action = mind.act(obs)
        self.assertTrue(0 <= action < 5)
        mind.update(obs, action, 1.0, next_obs, False)
        mind.update(next_obs, mind.act(next_obs), 2.0, obs, False)
        self.assertIsNotNone(mind.q_net)
        self.assertIsInstance(mind.q_net, StructuredQNet)
        self.assertTrue(0 <= mind.greedy_action(obs) < 5)

    def test_ppo_mind_outputs_valid_actions_and_updates(self):
        mind = PPOMind(action_dim=5, obs_dim=10, hidden_dim=8, rollout_steps=2, batch_size=2, seed=5)
        self.assertIsInstance(mind, TorchPPOMind)
        obs = (1, 2)
        next_obs = (2, 3)
        action = mind.act(obs)
        self.assertTrue(0 <= action < 5)
        mind.update(obs, action, 1.0, next_obs, False)
        mind.update(next_obs, mind.act(next_obs), 0.5, obs, False)
        self.assertIsInstance(mind.policy_net, StructuredPolicyNet)
        self.assertIsInstance(mind.value_net, StructuredValueNet)
        self.assertTrue(0 <= mind.greedy_action(obs) < 5)

    def test_numpy_fallback_minds_are_explicit_simple_baselines(self):
        self.assertIsInstance(SimpleDQNMind(action_dim=3, obs_dim=6, seed=1), SimpleDQNMind)
        self.assertIsInstance(SimplePPOMind(action_dim=3, obs_dim=6, seed=1), SimplePPOMind)


class MARLTests(unittest.TestCase):
    def test_independent_learners_coordinate_multiple_agents(self):
        learners = IndependentLearners(
            n_agents=2,
            mind="dqn",
            action_dim=5,
            obs_dim=10,
            hidden_dim=8,
            batch_size=2,
            min_replay_size=2,
            seed=7,
        )
        observations = [(1, 2), (1, 2)]
        actions = learners.act(observations)
        self.assertEqual(len(actions), 2)
        learners.update(observations, actions, [1.0, 2.0], [(2, 3), (2, 3)], [False, False])

    def test_centralized_critic_updates_from_joint_observation(self):
        learners = CentralizedCriticLearners(
            n_agents=2,
            action_dim=5,
            obs_dim=10,
            hidden_dim=8,
            rollout_steps=2,
            batch_size=2,
            seed=8,
        )
        observations = [(1, 2), (1, 2)]
        actions = [agent.act(obs) for agent, obs in zip(learners.agents, observations)]
        learners.update(observations, actions, [1.0, 2.0], [(2, 3), (2, 3)], [False, False])
        self.assertTrue(np.isfinite(learners.last_td_error))
        self.assertIsNotNone(learners.critic)


class Phase3PricingIntegrationTests(unittest.TestCase):
    def test_phase3_minds_run_pricing_arena_smoke(self):
        for mind in ("dqn", "ppo", "independent_dqn", "centralized_critic"):
            with self.subTest(mind=mind):
                data, _ = train_market(mechanism="none", steps=6, seed=3, mind=mind)
                self.assertEqual(len(data["avg_price"]), 6)
                self.assertTrue(np.all(np.isfinite(data["welfare"])))

    def test_phase3_validation_runner_writes_expected_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parse_args(
                [
                    "--minds",
                    "dqn",
                    "ppo",
                    "--best-response-steps",
                    "5",
                    "--smoke-steps",
                    "5",
                    "--save-dir",
                    tmpdir,
                ]
            )
            outputs = run_validation(args)
            self.assertTrue(outputs["best_response"].exists())
            self.assertTrue(outputs["pricing_smoke"].exists())
            with outputs["best_response"].open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["mind"] for row in rows}, {"dqn", "ppo"})


if __name__ == "__main__":
    unittest.main()
