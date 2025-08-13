import os
import json
import numpy as np
from django.test import TestCase
from .services.hopfield_core import HopfieldCore
from .services.chimera_agent import ChimeraAgent
from .services.gng_engine import GNG_Engine

class HopfieldCoreTest(TestCase):
    def test_learning_and_recall(self):
        core = HopfieldCore(dimensions=4)
        pattern1 = np.array([1, 1, -1, -1])
        pattern2 = np.array([1, -1, 1, -1])
        core.learn(pattern1)
        core.learn(pattern2)
        noisy_pattern1 = np.array([1, 1, -1, 1])
        recalled_pattern = core.recall(noisy_pattern1)
        self.assertTrue(np.array_equal(pattern1, recalled_pattern))

class GNGOptimizationTest(TestCase):
    def test_edge_utility_pruning(self):
        """Tests that low-utility edges are pruned."""
        gng = GNG_Engine(dimensions=2, n_iter_before_pruning=1, utility_prune_threshold=0.5)
        # Manually create a graph
        gng.nodes = {0: {'weight': np.array([0,0]), 'error': 0}, 1: {'weight': np.array([1,1]), 'error': 0}, 2: {'weight': np.array([2,2]), 'error': 0}}
        # Edge (0,1) has high utility, edge (1,2) has low utility
        gng.edges = {(0, 1, 0, 10), (1, 2, 0, 1)}

        # Process an input to trigger pruning
        gng.process_input(np.array([0.1, 0.1]))

        edge_endpoints = {(e[0], e[1]) for e in gng.edges}
        self.assertIn((0,1), edge_endpoints, "High-utility edge should remain.")
        self.assertNotIn((1,2), edge_endpoints, "Low-utility edge should be pruned.")

    def test_node_merging(self):
        """Tests that close nodes are merged."""
        gng = GNG_Engine(dimensions=2, n_iter_before_merging=1, merge_threshold=0.2)
        # Manually create a graph with two very close nodes (0 and 2)
        gng.nodes = {
            0: {'weight': np.array([0.0, 0.0]), 'error': 1},
            1: {'weight': np.array([5.0, 5.0]), 'error': 1},
            2: {'weight': np.array([0.1, 0.1]), 'error': 1}
        }
        gng.edges = {(0, 1, 0, 5)} # Edge connecting one of the close nodes
        initial_node_count = len(gng.nodes)

        # Process an input to trigger merging
        gng.process_input(np.array([2.0, 2.0]))

        self.assertEqual(len(gng.nodes), initial_node_count - 1, "Node count should decrease by 1 after merge.")

        # Check that the edge was re-routed to the new node
        new_node_id = max(gng.nodes.keys()) # The new node will have the highest ID
        edge_endpoints = {(e[0], e[1]) for e in gng.edges}
        self.assertIn(tuple(sorted((1, new_node_id))), edge_endpoints, "Edge should be re-routed to the new merged node.")


class CognitiveArchitectureServiceTest(TestCase):
    def setUp(self):
        self.agent_id = "test-agent-123"
        self.storage_path = os.path.join('backend', 'storage', f'{self.agent_id}.npz')
        if os.path.exists(self.storage_path): os.remove(self.storage_path)

    def tearDown(self):
        if os.path.exists(self.storage_path): os.remove(self.storage_path)

    def test_creation_and_persistence(self):
        self.assertFalse(os.path.exists(self.storage_path))
        agent = ChimeraAgent(agent_id=self.agent_id, load_from_storage=False)
        self.assertTrue(os.path.exists(self.storage_path))

    def test_hyperparameter_persistence(self):
        """Tests that GNG hyperparameters are saved and loaded."""
        hyperparams = {'merge_threshold': 0.77, 'utility_prune_threshold': 0.88}
        agent1 = ChimeraAgent(agent_id=self.agent_id, load_from_storage=False, **hyperparams)

        agent2 = ChimeraAgent(agent_id=self.agent_id, load_from_storage=True)
        # Check that the hyperparams were loaded into the GNG engine inside the STAG framework
        gng_engine = agent2.stag.tree['gng']
        self.assertEqual(gng_engine.merge_threshold, 0.77)
        self.assertEqual(gng_engine.utility_prune_threshold, 0.88)
