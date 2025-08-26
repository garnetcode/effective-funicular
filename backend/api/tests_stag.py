import os
import numpy as np
from django.test import TestCase
from unittest.mock import patch, MagicMock

import torch
import shutil
from .services.chimera_agent import ChimeraAgent
from .services.replay_buffer import Experience

class ChimeraAgentStagDecouplingTests(TestCase):
    def setUp(self):
        """Set up a ChimeraAgent instance for testing."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.agent_id = "test-stag-decoupling-agent"
        self.obs_dim = 10
        self.action_dim = 4
        self.pretrain_steps = 100

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
            hyperparams={
                'world_model_pretrain_steps': self.pretrain_steps,
                'stag_update_frequency': 10,
                'batch_size': 4,
                'imagine_horizon': 2,
                'sequence_length': 4
            }
        )
        self.agent.set_active_skill("test_skill")

    def tearDown(self):
        # Clean up the created agent history directory
        history_dir = self.agent.history_manager.storage_dir
        if os.path.exists(history_dir):
            shutil.rmtree(history_dir)

    def test_stag_pretraining_phase(self):
        """
        Tests that STAG is inactive during the world model pre-training phase.
        """
        # Set steps_done to be within the pre-training period
        self.agent.steps_done = self.pretrain_steps - 1
        state = np.random.rand(self.obs_dim)

        # 1. Test perceive_and_update_state
        # STAG should not be engaged, so activation_path should be empty.
        _, _, _, activation_path, novelty = self.agent.perceive_and_update_state("test_cortex", state)
        self.assertEqual(activation_path, [])
        self.assertEqual(novelty, 0)

        # 2. Test update_stag
        # The GNG's process_input method should not be called.
        active_stag = self.agent.skill_manager._get_or_create_stag(self.agent.active_skill_id)
        with patch.object(active_stag.tree['gng'], 'process_input') as mock_process_input:
            self.agent.update_stag(np.random.rand(self.agent.hidden_dim), 1.0)
            mock_process_input.assert_not_called()

        # 3. Test train_policy_in_imagination
        # The STAG context vector should be all zeros.
        with patch.object(self.agent.action_head.layer, 'forward', return_value=torch.randn(1, self.action_dim)) as mock_forward:
            # We need to simulate the imagination loop to check the input to the action head
            # To do this cleanly, we can check the logic within train_policy_in_imagination
            # For simplicity, we'll mock the whole function and trust the internal logic from inspection.
            # A more complex test could mock `find_terminal_node_and_path` to return a specific path
            # and check the resulting context vector.
            # Here, we check that if we call it, the stag context part of the input to the action head is zero.
            # This is implicitly tested by the fact that activation_path is empty.
            # Let's check the context generation during imagination explicitly.
            with patch.object(self.agent, 'replay_buffer') as mock_buffer:
                batch_size = self.agent.hyperparams.get('batch_size', 4)

                # Create a list of valid Experience tuples
                mock_experiences = []
                for _ in range(batch_size):
                    exp = Experience(
                        h=torch.rand(1, self.agent.hidden_dim),
                        z=torch.rand(1, self.agent.latent_dim),
                        activation_path=[],
                        obs=np.zeros(self.obs_dim),
                        action=0,
                        log_prob=0.5,
                        reward=0.0,
                        next_obs=np.zeros(self.obs_dim),
                        done=False,
                        goal=np.zeros(self.obs_dim)
                    )
                    mock_experiences.append(exp)

                mock_buffer.sample.return_value = (
                    {
                        'h': np.random.rand(batch_size, 1, self.agent.hidden_dim),
                        'z': np.random.rand(batch_size, 1, self.agent.latent_dim),
                        'goal': np.random.rand(batch_size, 1, self.agent.goal_dim),
                    },
                    None, None
                )
                mock_buffer.__len__.return_value = batch_size
                mock_buffer.sequence_length = self.agent.hyperparams.get('sequence_length')

                # Patch the action head's forward method to capture the input
                with patch.object(self.agent.action_head.layer, 'forward') as mock_action_layer_forward:
                    # The mock needs to return a valid tensor for the Categorical distribution
                    mock_action_layer_forward.return_value = torch.randn(batch_size, self.agent.max_action_dim)

                    self.agent.train_policy_in_imagination()
                    # Get the input to the action head from the first call in the imagination loop
                    call_args, _ = mock_action_layer_forward.call_args
                    combined_input = call_args[0]
                    stag_context_part = combined_input[:, self.agent.hidden_dim:]
                    self.assertTrue(torch.all(stag_context_part == 0))


    def test_stag_activation_post_pretraining(self):
        """
        Tests that STAG becomes active after the pre-training phase.
        """
        # Set steps_done to be after the pre-training period
        self.agent.steps_done = self.pretrain_steps + 1
        state = np.random.rand(self.obs_dim)

        # 1. Test perceive_and_update_state
        # STAG should now be engaged, so activation_path should not be empty.
        _, _, h_normalized, activation_path, _ = self.agent.perceive_and_update_state("test_cortex", state)
        self.assertNotEqual(activation_path, [])

        # 2. Test update_stag frequency
        active_stag = self.agent.skill_manager._get_or_create_stag(self.agent.active_skill_id)
        with patch.object(active_stag.tree['gng'], 'process_input') as mock_process_input:
            # Should not be called because steps_done % frequency != 0
            self.agent.steps_done = self.pretrain_steps + 1
            self.agent.update_stag(h_normalized, 1.0)
            mock_process_input.assert_not_called()

            # Should be called because steps_done % frequency == 0
            self.agent.steps_done = self.pretrain_steps + 10 # 110 % 10 == 0
            self.agent.update_stag(h_normalized, 1.0)
            mock_process_input.assert_called_once_with(h_normalized, reward=1.0)
