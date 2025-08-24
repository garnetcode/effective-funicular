import unittest
import numpy as np
import gymnasium as gym

from .factory import create_cortex_configs_from_observation_space

class CortexFactoryTests(unittest.TestCase):

    def test_vector_observation_space(self):
        """
        Tests that a 1D Box space is correctly identified as a vector input
        and configured for a DenseCortex.
        """
        # Create a mock vector observation space
        vector_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

        configs, cortex_id = create_cortex_configs_from_observation_space(vector_space)

        self.assertEqual(cortex_id, "vector_cortex")
        self.assertIn("vector_cortex", configs)
        self.assertEqual(configs["vector_cortex"]["type"], "DenseCortex")
        self.assertEqual(configs["vector_cortex"]["params"]["input_dim"], 10)

    def test_image_observation_space(self):
        """
        Tests that a 3D Box space is correctly identified as an image input
        and configured for a VisionCortex.
        """
        # Create a mock image observation space
        image_shape = (84, 84, 3)
        image_space = gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)

        configs, cortex_id = create_cortex_configs_from_observation_space(image_space)

        self.assertEqual(cortex_id, "vision_cortex")
        self.assertIn("vision_cortex", configs)
        self.assertEqual(configs["vision_cortex"]["type"], "VisionCortex")
        self.assertEqual(configs["vision_cortex"]["params"]["input_shape"], image_shape)

    def test_unsupported_observation_space(self):
        """
        Tests that an unsupported observation space raises a NotImplementedError.
        """
        # Create a mock unsupported space (e.g., a 2D Box space)
        unsupported_space = gym.spaces.Box(low=0, high=1, shape=(10, 10), dtype=np.float32)

        with self.assertRaises(NotImplementedError):
            create_cortex_configs_from_observation_space(unsupported_space)

if __name__ == '__main__':
    unittest.main()
