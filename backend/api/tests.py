import os
import numpy as np
from django.test import TestCase
from unittest.mock import patch, MagicMock

import torch
import shutil
from .services.gng_engine import GNG_Engine
from .services.chimera_agent import ChimeraAgent
from .services.state_history_manager import StateHistoryManager

class GNG_EngineTests(TestCase):
    def setUp(self):
        # We use a fixed seed for reproducibility of random initial nodes
        np.random.seed(42)
        self.gng = GNG_Engine(
            dimensions=2,
            gng_utility_gain=1.0,
            gng_utility_decay_rate=0.0,
            gng_pruning_grace_period=0
        )

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
        # AGENT_FIX: The GNG now uses 'error_fast' and 'error_slow'. The test
        # should reflect this change. Using 'error_slow' as it's used for insertion logic.
        q_id = max(self.gng.nodes, key=lambda nid: self.gng.nodes[nid]['error_slow'])
        q_neighbors = self.gng._get_neighbors(q_id)
        # Manually create a neighbor if one doesn't exist for the test
        if not q_neighbors:
            other_node_id = next(nid for nid in self.gng.nodes if nid != q_id)
            self.gng.edges.add(tuple(sorted((q_id, other_node_id))) + (0,))
            q_neighbors = self.gng._get_neighbors(q_id)

        f_id = max(q_neighbors, key=lambda nid: self.gng.nodes[nid]['error_slow'])

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

    def test_prune_low_utility_nodes(self):
        """Tests that nodes with low utility are correctly pruned."""
        # Add more nodes to the GNG
        node2_id = self.gng._add_node(np.random.randn(2))
        node3_id = self.gng._add_node(np.random.randn(2))
        node4_id = self.gng._add_node(np.random.randn(2))

        # Create edges
        self.gng.edges.add(tuple(sorted((0, 1))) + (0,))
        self.gng.edges.add(tuple(sorted((1, node2_id))) + (0,))
        self.gng.edges.add(tuple(sorted((node2_id, node3_id))) + (0,))
        self.gng.edges.add(tuple(sorted((node3_id, node4_id))) + (0,))

        # Set utilities
        self.gng.nodes[0]['utility'] = 5.0  # High utility
        self.gng.nodes[1]['utility'] = 0.05 # Low utility (to be pruned)
        self.gng.nodes[node2_id]['utility'] = 6.0  # High utility
        self.gng.nodes[node3_id]['utility'] = 0.01 # Low utility (to be pruned)
        self.gng.nodes[node4_id]['utility'] = 7.0  # High utility

        min_utility_threshold = 0.1
        self.gng._iterations = 1 # Ensure nodes are not in their grace period
        self.gng.prune_low_utility_nodes(min_utility_threshold)

        # Assert that low-utility nodes are gone
        self.assertNotIn(1, self.gng.nodes)
        self.assertNotIn(node3_id, self.gng.nodes)

        # Assert that high-utility nodes remain
        self.assertIn(0, self.gng.nodes)
        self.assertIn(node2_id, self.gng.nodes)
        self.assertIn(node4_id, self.gng.nodes)
        self.assertEqual(len(self.gng.nodes), 3)

        # Assert that edges connected to pruned nodes are gone
        remaining_edges = {e[:2] for e in self.gng.edges}
        self.assertNotIn(tuple(sorted((0, 1))), remaining_edges)
        self.assertNotIn(tuple(sorted((1, node2_id))), remaining_edges)
        self.assertNotIn(tuple(sorted((node2_id, node3_id))), remaining_edges)
        self.assertNotIn(tuple(sorted((node3_id, node4_id))), remaining_edges)

        # The only remaining nodes are 0, 2, 4, which are not connected in this test setup.
        self.assertEqual(len(self.gng.edges), 0)

    def test_two_phase_learning_rate(self):
        """Tests that the GNG correctly switches from ordering to tuning learning rates."""
        gng_2phase = GNG_Engine(
            dimensions=2,
            gng_ordering_phase_steps=10,
            gng_winner_learning_rate_initial=0.8,
            gng_winner_learning_rate=0.1 # This is the tuning rate
        )
        input_vector = np.array([0.5, 0.5])

        # --- Test Ordering Phase ---
        gng_2phase._iterations = 5 # Well within the ordering phase
        winner_id, _ = gng_2phase._find_winners(input_vector)
        initial_weight = gng_2phase.nodes[winner_id]['weight'].copy()
        gng_2phase.nodes[winner_id]['utility'] = 1.0 # Set utility to a known value

        gng_2phase.process_input(input_vector)
        ordering_weight_change = np.linalg.norm(gng_2phase.nodes[winner_id]['weight'] - initial_weight)

        # --- Test Tuning Phase ---
        gng_2phase._iterations = 15 # Past the ordering phase
        # Reset the weight and utility to have a fair comparison
        gng_2phase.nodes[winner_id]['weight'] = initial_weight
        gng_2phase.nodes[winner_id]['utility'] = 1.0

        gng_2phase.process_input(input_vector)
        tuning_weight_change = np.linalg.norm(gng_2phase.nodes[winner_id]['weight'] - initial_weight)

        # The weight change should be larger during the high-learning-rate ordering phase
        self.assertGreater(ordering_weight_change, tuning_weight_change)


class ChimeraAgentTests(TestCase):
    def setUp(self):
        """Set up a ChimeraAgent instance for testing with the new WorldModel."""
        torch.manual_seed(42)
        self.agent_id = "test-world-model-agent"
        self.obs_dim = 10
        self.action_dim = 4

        # Mock cortex config
        cortex_configs = {
            "test_cortex": {
                "type": "DenseCortex",
                "params": {"input_dim": self.obs_dim}
            }
        }

        self.agent = ChimeraAgent(
            agent_id=self.agent_id,
            embedding_dim=self.obs_dim,
            max_action_dim=self.action_dim,
            cortex_configs=cortex_configs,
            load_from_storage=False,
            hyperparams={'batch_size': 16}
        )
        # Ensure a clean state for each test
        self.agent.episode_memory = []
        self.agent.set_active_skill("test_skill")

    def tearDown(self):
        # Clean up the created agent history directory
        history_dir = self.agent.history_manager.storage_dir
        if os.path.exists(history_dir):
            shutil.rmtree(history_dir)

    def test_online_learning_loop(self):
        """
        Tests the agent's ability to record experiences and improve its
        world model through online, batch-based training.
        """
        # 1. Populate the replay buffer with enough experiences to start training
        # The number of experiences must be >= sequence_length
        sequence_length = self.agent.hyperparams.get('sequence_length', 50)
        for _ in range(sequence_length + 5):
            # h, z, activation_path, obs, action, log_prob, reward, next_obs, done
            self.agent.record_experience(
                torch.rand(1, self.agent.hidden_dim), # h
                torch.rand(1, self.agent.latent_dim), # z
                [],                                   # activation_path
                np.random.rand(self.obs_dim),         # obs
                np.random.randint(0, self.action_dim),# action
                0.5,                                  # log_prob
                1.0,                                  # reward
                np.random.rand(self.obs_dim),         # next_obs
                False                                 # done
            )

        self.assertEqual(len(self.agent.replay_buffer), sequence_length + 5)

        # 2. Run an initial training step and record the loss
        initial_train_stats = self.agent.train(cortex_id="test_cortex")
        initial_loss = initial_train_stats.get("wm_loss_total")
        self.assertIsNotNone(initial_loss)

        # 3. Run several more training steps
        final_loss = initial_loss
        for i in range(5):
            # Add one new experience
            self.agent.record_experience(
                torch.rand(1, self.agent.hidden_dim), # h
                torch.rand(1, self.agent.latent_dim), # z
                [],                                   # activation_path
                np.random.rand(self.obs_dim),         # obs
                np.random.randint(0, self.action_dim),# action
                0.5,                                  # log_prob
                1.0,                                  # reward
                np.random.rand(self.obs_dim),         # next_obs
                False                                 # done
            )
            # Train on a new batch
            train_stats = self.agent.train(cortex_id="test_cortex")
            # The loss might be None if the buffer is not full enough, which is ok.
            if train_stats and train_stats.get("wm_loss_total") is not None:
                final_loss = train_stats.get("wm_loss_total")

        # 4. Assert that the loss is a valid number (it's not guaranteed to decrease every step)
        self.assertIsInstance(final_loss, float)

    @patch('api.services.state_history_manager.StateHistoryManager.save_snapshot')
    def test_world_model_weight_decay(self, mock_save_snapshot):
        """Tests that the weight_decay hyperparameter is correctly passed to the optimizer."""
        decay_value = 0.01

        with patch('torch.optim.Adam') as mock_adam:
            agent = ChimeraAgent(
                agent_id="test-weight-decay-agent",
                embedding_dim=self.obs_dim,
                max_action_dim=self.action_dim,
                cortex_configs=self.agent.cortex_configs,
                load_from_storage=False,
                hyperparams={'world_model_weight_decay': decay_value, 'batch_size': 1}
            )
            # Check if Adam was called during initialization
            self.assertTrue(mock_adam.called)

            # Check the keyword arguments passed to Adam
            # It's called for each model in the ensemble
            for call in mock_adam.call_args_list:
                _, kwargs = call
                self.assertIn('weight_decay', kwargs)
                self.assertEqual(kwargs['weight_decay'], decay_value)

        # Clean up the agent's directory
        history_dir = agent.history_manager.storage_dir
        if os.path.exists(history_dir):
            shutil.rmtree(history_dir)


class StateHistoryManagerTests(TestCase):
    def setUp(self):
        self.agent_id = "test-history-agent"
        self.storage_root = "backend/test_storage"
        # Override max_snapshots to a higher value for this specific test
        # to prevent pruning from interfering with the test logic.
        self.manager = StateHistoryManager(self.agent_id, storage_root=self.storage_root, base_snapshot_interval=3, max_snapshots=10)

    def tearDown(self):
        # Clean up the created history directory
        if os.path.exists(self.manager.storage_dir):
            shutil.rmtree(self.manager.storage_dir)

    def test_save_and_load_versioning(self):
        """
        Tests the full base-and-diff versioning and reconstruction logic.
        """
        # Create a series of states
        states = []
        state_v0 = {'param1': torch.tensor([1.0, 2.0]), 'param2': torch.tensor([3.0])}
        states.append(state_v0)

        # Version 0 (base)
        self.manager.save_snapshot(states[0], version_info={'loss': 10.0})

        # Version 1 (diff)
        states.append({'param1': torch.tensor([1.5, 2.5]), 'param2': torch.tensor([3.5])})
        self.manager.save_snapshot(states[1], version_info={'loss': 9.0})

        # Version 2 (diff)
        states.append({'param1': torch.tensor([1.5, 3.0]), 'param2': torch.tensor([4.0])})
        self.manager.save_snapshot(states[2], version_info={'loss': 8.0})

        # Version 3 (base)
        states.append({'param1': torch.tensor([2.0, 2.0]), 'param2': torch.tensor([2.0])})
        self.manager.save_snapshot(states[3], version_info={'loss': 7.0})

        # Version 4 (diff)
        states.append({'param1': torch.tensor([2.5, 2.5]), 'param2': torch.tensor([2.5])})
        self.manager.save_snapshot(states[4], version_info={'loss': 6.0})

        # --- Test Reconstruction ---

        # Test loading the latest version (v4)
        loaded_v4 = self.manager.load_snapshot('latest')
        self.assertTrue(torch.allclose(loaded_v4['param1'], states[4]['param1']))
        self.assertTrue(torch.allclose(loaded_v4['param2'], states[4]['param2']))

        # Test loading an intermediate diff version (v2)
        loaded_v2 = self.manager.load_snapshot(2)
        self.assertTrue(torch.allclose(loaded_v2['param1'], states[2]['param1']))
        self.assertTrue(torch.allclose(loaded_v2['param2'], states[2]['param2']))

        # Test loading a base version (v3)
        loaded_v3 = self.manager.load_snapshot(3)
        self.assertTrue(torch.allclose(loaded_v3['param1'], states[3]['param1']))
        self.assertTrue(torch.allclose(loaded_v3['param2'], states[3]['param2']))

        # Test history log
        history = self.manager._read_history()
        self.assertEqual(len(history), 5)
        self.assertEqual(history[0]['type'], 'base')
        self.assertEqual(history[1]['type'], 'diff')
        self.assertEqual(history[2]['type'], 'diff')
        self.assertEqual(history[3]['type'], 'base')
        self.assertEqual(history[4]['type'], 'diff')
        self.assertEqual(history[4]['info']['loss'], 6.0)
