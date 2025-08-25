import unittest
import numpy as np
import torch
import shutil

from .chimera_agent import ChimeraAgent

class SkillIsolationTests(unittest.TestCase):
    def setUp(self):
        self.agent_id = "test-skill-isolation-agent"
        self.embedding_dim = 16
        self.action_dim = 4
        self.hidden_dim = 32
        self.latent_dim = 16

        self.agent = ChimeraAgent(
            agent_id=self.agent_id,
            embedding_dim=self.embedding_dim,
            max_action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            load_from_storage=False,
            hyperparams={'world_model_pretrain_steps': 0} # Ensure STAG is active
        )

    def tearDown(self):
        history_dir = self.agent.history_manager.storage_dir
        if shutil.os.path.exists(history_dir):
            shutil.rmtree(history_dir)

    def test_skill_graphs_are_isolated(self):
        """
        Tests that training on one skill does not affect the graph of another.
        """
        # --- Train on Skill A ---
        skill_a_id = "CartPole-v1"
        self.agent.set_active_skill(skill_a_id)

        # Process some data for skill A to create a graph
        for _ in range(5):
            h_normalized = np.random.randn(self.hidden_dim)
            self.agent.update_stag(h_normalized, r_env=1.0)

        graph_a_before = self.agent.get_graph_structure(skill_id=skill_a_id)
        self.assertTrue(len(graph_a_before['nodes']) > 0, "Graph A should have nodes after training.")

        # --- Train on Skill B ---
        skill_b_id = "LunarLander-v2"
        self.agent.set_active_skill(skill_b_id)

        # Process some data for skill B
        for _ in range(5):
            h_normalized = np.random.randn(self.hidden_dim)
            self.agent.update_stag(h_normalized, r_env=1.0)

        graph_b = self.agent.get_graph_structure(skill_id=skill_b_id)
        self.assertTrue(len(graph_b['nodes']) > 0, "Graph B should have nodes after training.")

        # --- Verify Skill A is unchanged ---
        graph_a_after = self.agent.get_graph_structure(skill_id=skill_a_id)

        # The number of nodes and edges should be identical
        self.assertEqual(len(graph_a_before['nodes']), len(graph_a_after['nodes']))
        self.assertEqual(len(graph_a_before['edges']), len(graph_a_after['edges']))

        # The weights of the nodes should be identical
        for node_id, node_data_before in graph_a_before['nodes'].items():
            node_data_after = graph_a_after['nodes'][node_id]
            np.testing.assert_array_equal(node_data_before['weight'], node_data_after['weight'])

if __name__ == '__main__':
    unittest.main()
