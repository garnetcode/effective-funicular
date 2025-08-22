import numpy as np
from django.test import TestCase
from unittest.mock import patch, MagicMock

import torch
from .services.gng_engine import GNG_Engine
from .services.chimera_agent import ChimeraAgent
from .services.hopfield_rnn_core import StableHopfieldRNN

class GNG_EngineTests(TestCase):
    def setUp(self):
        # We use a fixed seed for reproducibility of random initial nodes
        np.random.seed(42)
        self.gng = GNG_Engine(dimensions=2, gng_utility_gain=1.0, gng_utility_decay_rate=0.0)

    def test_utility_gain_on_winner(self):
        """Test that the winning node's utility increases after processing an input."""
        initial_utilities = {nid: node['utility'] for nid, node in self.gng.nodes.items()}
        input_vector = np.array([0.1, 0.1])

        winner_id, _ = self.gng._find_winners(input_vector)
        self.gng.process_input(input_vector)

        # The winner's utility should increase by gng_utility_gain (1.0)
        self.assertAlmostEqual(self.gng.nodes[winner_id]['utility'], initial_utilities[winner_id] + 1.0)

    def test_dynamic_learning_rate(self):
        """Test that the effective learning rate decreases as utility increases."""
        input_vector = np.array([0.5, 0.5])
        winner_id, _ = self.gng._find_winners(input_vector)

        # First pass: utility is low, learning should be high
        initial_weight = self.gng.nodes[winner_id]['weight'].copy()
        self.gng.process_input(input_vector)
        first_pass_weight_change = np.linalg.norm(self.gng.nodes[winner_id]['weight'] - initial_weight)

        # Process the same input multiple times to increase utility
        for _ in range(10):
            self.gng.process_input(input_vector)

        self.assertTrue(self.gng.nodes[winner_id]['utility'] > 2.0) # Utility should be significantly higher

        # Second pass: utility is high, learning should be lower
        second_pass_initial_weight = self.gng.nodes[winner_id]['weight'].copy()
        self.gng.process_input(input_vector)
        second_pass_weight_change = np.linalg.norm(self.gng.nodes[winner_id]['weight'] - second_pass_initial_weight)

        self.assertLess(second_pass_weight_change, first_pass_weight_change)

    def test_node_insertion_utility_inheritance(self):
        """Test that a new node inherits the average utility of its parents."""
        self.gng.n_iter_before_neuron_added = 1

        # Find nodes with highest error to predict insertion point
        q_id = max(self.gng.nodes, key=lambda nid: self.gng.nodes[nid]['error'])
        q_neighbors = self.gng._get_neighbors(q_id)
        # Manually create a neighbor if one doesn't exist for the test
        if not q_neighbors:
            other_node_id = next(nid for nid in self.gng.nodes if nid != q_id)
            self.gng.edges.add(tuple(sorted((q_id, other_node_id))) + (0,))
            q_neighbors = self.gng._get_neighbors(q_id)

        f_id = max(q_neighbors, key=lambda nid: self.gng.nodes[nid]['error'])

        # Set known utilities for parents
        self.gng.nodes[q_id]['utility'] = 10.0
        self.gng.nodes[f_id]['utility'] = 20.0
        expected_utility = 15.0

        # Trigger insertion
        self.gng._insert_node()

        # The new node should be the one with the highest ID
        new_node_id = self.gng._next_node_id - 1
        self.assertIn(new_node_id, self.gng.nodes)
        self.assertAlmostEqual(self.gng.nodes[new_node_id]['utility'], expected_utility)


class ChimeraAgentTests(TestCase):
    def setUp(self):
        # Use a unique agent_id for test isolation
        self.agent_id = "test-consolidation-agent"
        # Create a real agent instance. This avoids mock serialization issues.
        self.agent = ChimeraAgent(agent_id=self.agent_id, load_from_storage=False, dimensions=8)
        self.agent.patterns = {0: np.random.rand(8), 1: np.random.rand(8)}
        self.agent.save_state()

    def test_consolidation_calls_organize_memory(self):
        """Test that consolidate_memories calls organize_memory for each pattern."""
        n_replays = 3

        # We patch 'organize_memory' on the instance for just this test.
        with patch.object(self.agent, 'organize_memory', return_value=None) as mock_organize_memory:
            self.agent.consolidate_memories(n_replays=n_replays)

            # Check that organize_memory was called the correct number of times
            self.assertEqual(mock_organize_memory.call_count, len(self.agent.patterns) * n_replays)

            # Check that it was called with the correct pattern IDs
            called_pattern_ids = [call.args[0] for call in mock_organize_memory.call_args_list]
            for pattern_id in self.agent.patterns.keys():
                self.assertIn(pattern_id, called_pattern_ids)

    def tearDown(self):
        # Clean up the created agent file
        import os
        storage_path = os.path.join('backend', 'storage', f'{self.agent_id}.npz')
        if os.path.exists(storage_path):
            os.remove(storage_path)

    def test_homeostatic_vitals_and_reward(self):
        """
        Test that agent vitals are updated correctly and influence the reward.
        """
        # 1. Check initial state
        self.assertEqual(self.agent.energy, self.agent.max_energy)
        self.assertEqual(self.agent.integrity, 100.0)

        # 2. Simulate a basic step with metabolic cost
        old_energy = self.agent.energy
        # Manually apply metabolic cost
        self.agent.energy -= self.agent.metabolic_cost
        homeostatic_reward = self.agent.energy - old_energy
        # Record experience with only homeostatic reward for this test
        self.agent.record_experience(np.random.rand(8), 0, 0.5, homeostatic_reward)

        self.assertAlmostEqual(self.agent.energy, self.agent.max_energy - self.agent.metabolic_cost)
        # The reward recorded should be the negative metabolic cost
        self.assertAlmostEqual(self.agent.episode_memory[0]['reward'], -self.agent.metabolic_cost)
        self.agent.episode_memory = [] # Clear memory

        # 3. Simulate finding food (energy gain)
        old_energy = self.agent.energy
        energy_gain = 20.0
        self.agent.energy += energy_gain
        homeostatic_reward = self.agent.energy - old_energy
        self.agent.record_experience(np.random.rand(8), 1, 0.5, homeostatic_reward)

        self.assertAlmostEqual(self.agent.energy, self.agent.max_energy - self.agent.metabolic_cost + energy_gain)
        self.assertAlmostEqual(self.agent.episode_memory[0]['reward'], energy_gain)
        self.agent.episode_memory = []

        # 4. Simulate taking damage (integrity loss)
        old_integrity = self.agent.integrity
        integrity_loss = -10.0
        self.agent.integrity += integrity_loss
        homeostatic_reward = self.agent.integrity - old_integrity
        self.agent.record_experience(np.random.rand(8), 2, 0.5, homeostatic_reward)

        self.assertAlmostEqual(self.agent.integrity, 100.0 + integrity_loss)
        self.assertAlmostEqual(self.agent.episode_memory[0]['reward'], integrity_loss)


class StableHopfieldRNNTests(TestCase):
    def setUp(self):
        """Set up a StableHopfieldRNN instance for testing."""
        self.num_nodes = 4
        self.hidden_dim = 8
        torch.manual_seed(42)
        self.hrnn = StableHopfieldRNN(num_nodes=self.num_nodes, hidden_dim=self.hidden_dim, alpha=0.1)

    def test_training_and_convergence(self):
        """
        Test that the network can learn a pattern and recall it from a noisy cue.
        """
        # 1. Define a clear pattern to be memorized
        # Use tanh to ensure values are in [-1, 1], typical for hidden states
        pattern_to_memorize = torch.tanh(torch.randn(self.num_nodes, self.hidden_dim))

        # 2. Train the network on this pattern
        self.hrnn.train_on_pattern(pattern_to_memorize, num_epochs=100, learning_rate=0.01, fixed_point_lambda=0.5)

        # 3. Create a noisy version of the pattern to use as a cue
        noise = torch.randn(self.num_nodes, self.hidden_dim) * 0.5
        noisy_cue = pattern_to_memorize + noise

        # 4. Perform the recall process
        recalled_pattern = self.hrnn.forward(noisy_cue, num_updates=50)

        # 5. Check for convergence
        # The recalled pattern should be closer to the original than the noisy cue was.
        initial_distance = torch.norm(pattern_to_memorize - noisy_cue)
        final_distance = torch.norm(pattern_to_memorize - recalled_pattern)

        print(f"Initial distance: {initial_distance.item()}, Final distance: {final_distance.item()}")

        self.assertLess(final_distance, initial_distance)
