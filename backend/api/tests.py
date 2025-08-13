import os
import json
import numpy as np
from django.test import TestCase
from .services.hopfield_core import HopfieldCore
from .services.chimera_agent import ChimeraAgent
from .services.gng_engine import GNG_Engine

# NOTE: These tests require faiss to be installed.
try:
    import faiss
except ImportError:
    faiss = None

class GNGFaissTest(TestCase):
    def setUp(self):
        if not faiss:
            self.skipTest("FAISS library not installed, skipping tests.")

    def test_find_winners_correctness(self):
        """Tests that FAISS-based search returns the same winners as numpy."""
        gng = GNG_Engine(dimensions=4)

        # Manually create nodes
        nodes_data = {
            0: np.array([1, 1, 1, 1]),
            1: np.array([-1, -1, -1, -1]),
            2: np.array([5, 5, 5, 5]),
            3: np.array([1, 1, 1, 2]),
        }
        gng.nodes = {nid: {'weight': w, 'error': 0} for nid, w in nodes_data.items()}
        gng._rebuild_faiss_index() # Manually build the index from the nodes

        query_vector = np.array([1.1, 1.1, 1.1, 1.1])

        # Get winners from FAISS
        s1_faiss, s2_faiss = gng._find_winners(query_vector)

        # Get winners from numpy
        dists = {nid: np.linalg.norm(w - query_vector) for nid, w in nodes_data.items()}
        sorted_dists = sorted(dists.items(), key=lambda item: item[1])
        s1_numpy, s2_numpy = sorted_dists[0][0], sorted_dists[1][0]

        self.assertEqual(s1_faiss, s1_numpy, "The first winner should be the same.")
        self.assertEqual(s2_faiss, s2_numpy, "The second winner should be the same.")

class AgentPersistenceTest(TestCase):
    def setUp(self):
        self.agent_id = "test-agent-faiss-123"
        self.storage_path = os.path.join('backend', 'storage', f'{self.agent_id}.npz')
        self.faiss_path = os.path.join('backend', 'storage', f'{self.agent_id}.faiss')
        # Clean up previous runs
        if os.path.exists(self.storage_path): os.remove(self.storage_path)
        if os.path.exists(self.faiss_path): os.remove(self.faiss_path)

    def tearDown(self):
        if os.path.exists(self.storage_path): os.remove(self.storage_path)
        if os.path.exists(self.faiss_path): os.remove(self.faiss_path)

    def test_faiss_index_persistence(self):
        """Tests that the agent saves and loads its FAISS index correctly."""
        if not faiss:
            self.skipTest("FAISS library not installed.")

        # 1. Create and save an agent
        agent1 = ChimeraAgent(agent_id=self.agent_id, dimensions=4, load_from_storage=False)
        agent1.stag.tree['gng'].process_input(np.random.rand(4)) # Add some nodes
        agent1.save_state()

        self.assertTrue(os.path.exists(self.storage_path))
        self.assertTrue(os.path.exists(self.faiss_path))

        # 2. Load the agent
        agent2 = ChimeraAgent(agent_id=self.agent_id, load_from_storage=True)
        gng_engine = agent2.stag.tree['gng']

        self.assertIsNotNone(gng_engine.faiss_index, "FAISS index should be loaded.")
        self.assertEqual(gng_engine.faiss_index.ntotal, len(gng_engine.nodes), "Loaded FAISS index should have the correct number of vectors.")

        # 3. Verify it's functional
        s1, s2 = gng_engine._find_winners(np.random.rand(4))
        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)
