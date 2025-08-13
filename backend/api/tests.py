import os
import json
import numpy as np
from django.test import TestCase
from .services.hopfield_core import HopfieldCore
from .services.cognitive_architecture_service import CognitiveArchitectureService, text_to_embedding

class HopfieldCoreTest(TestCase):
    def test_learning_and_recall(self):
        dimensions = 4
        core = HopfieldCore(dimensions=dimensions)

        # Two simple, orthogonal patterns
        pattern1 = np.array([1, 1, -1, -1])
        pattern2 = np.array([1, -1, 1, -1])

        core.learn(pattern1)
        core.learn(pattern2)

        # Test recall with a noisy version of pattern1
        noisy_pattern1 = np.array([1, 1, -1, 1])
        recalled_pattern = core.recall(noisy_pattern1)

        self.assertTrue(np.array_equal(pattern1, recalled_pattern), "Should recall the correct pattern")

class CognitiveArchitectureServiceTest(TestCase):
    def setUp(self):
        self.network_id = "test-network-123"
        self.storage_path = os.path.join('backend', 'storage', f'{self.network_id}.json')

        # Clean up any old test files before starting
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def tearDown(self):
        # Clean up test files after finishing
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def test_creation_and_persistence(self):
        """Tests if the service creates a state file on initialization."""
        self.assertFalse(os.path.exists(self.storage_path))
        service = CognitiveArchitectureService(network_id=self.network_id, load_from_storage=False)
        self.assertTrue(os.path.exists(self.storage_path))

    def test_learn_and_save(self):
        """Tests if learning a pattern persists the state."""
        service = CognitiveArchitectureService(network_id=self.network_id, load_from_storage=False)

        initial_patterns_count = len(service.patterns)
        initial_weights = np.copy(service.hopfield.weights)

        service.learn_pattern("hello world")

        self.assertEqual(len(service.patterns), initial_patterns_count + 1)
        self.assertFalse(np.array_equal(initial_weights, service.hopfield.weights))

        # Verify by loading from file
        service2 = CognitiveArchitectureService(network_id=self.network_id, load_from_storage=True)
        self.assertEqual(len(service2.patterns), initial_patterns_count + 1)
        self.assertTrue(np.array_equal(service.hopfield.weights, service2.hopfield.weights))

    def test_organize_and_save(self):
        """Tests if organizing the network persists the state."""
        service = CognitiveArchitectureService(network_id=self.network_id, dimensions=4, load_from_storage=False)
        service.learn_pattern("pattern a")
        service.learn_pattern("pattern b")

        initial_stag_state = service.stag.get_serializable_structure()

        service.organize_step()

        final_stag_state = service.stag.get_serializable_structure()

        # The organization step should change the GNG state (e.g., iterations, errors)
        self.assertNotEqual(initial_stag_state, final_stag_state)

        # Verify by loading from file
        service2 = CognitiveArchitectureService(network_id=self.network_id, load_from_storage=True)
        self.assertEqual(final_stag_state, service2.stag.get_serializable_structure())
