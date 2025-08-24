import os
import numpy as np
from django.test import TestCase
from unittest.mock import patch, MagicMock

import torch
import shutil
from .services.chimera_agent import ChimeraAgent

class ChimeraAgentStagTests(TestCase):
    def setUp(self):
        """Set up a ChimeraAgent instance for testing."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.agent_id_stag_enabled = "test-stag-enabled-agent"
        self.agent_id_stag_disabled = "test-stag-disabled-agent"
        self.obs_dim = 10
        self.action_dim = 4

        cortex_configs = {
            "test_cortex": {
                "type": "DenseCortex",
                "params": {"input_dim": self.obs_dim}
            }
        }

        # Agent with STAG enabled (default behavior)
        self.agent_stag_enabled = ChimeraAgent(
            agent_id=self.agent_id_stag_enabled,
            max_obs_dim=self.obs_dim,
            max_action_dim=self.action_dim,
            cortex_configs=cortex_configs,
            load_from_storage=False,
            hyperparams={'use_stag_in_ac_loss': True}
        )

        # Agent with STAG disabled
        self.agent_stag_disabled = ChimeraAgent(
            agent_id=self.agent_id_stag_disabled,
            max_obs_dim=self.obs_dim,
            max_action_dim=self.action_dim,
            cortex_configs=cortex_configs,
            load_from_storage=False,
            hyperparams={'use_stag_in_ac_loss': False}
        )

    def tearDown(self):
        # Clean up the created agent history directories
        for agent in [self.agent_stag_enabled, self.agent_stag_disabled]:
            history_dir = agent.history_manager.storage_dir
            if os.path.exists(history_dir):
                shutil.rmtree(history_dir)

    @patch('api.services.chimera_agent.ChimeraAgent.train_policy_in_imagination')
    def test_stag_influence_toggle(self, mock_train_policy):
        """
        Tests that the `use_stag_in_ac_loss` flag correctly toggles
        the STAG context vector's influence on the action head.
        """
        # We patch `train_policy_in_imagination` because we are interested in the inputs to it,
        # which are determined by `select_action`.

        # --- Test STAG Enabled ---
        state = np.random.rand(self.obs_dim)
        _, _, _, activation_path, _ = self.agent_stag_enabled.perceive_and_update_state("test_cortex", state)

        with patch.object(self.agent_stag_enabled.action_head.layer, 'forward', return_value=torch.randn(1, self.action_dim)) as mock_action_head_forward_enabled:
            self.agent_stag_enabled.select_action(self.action_dim, activation_path)
            # The input to the action head's forward method is the combined_input
            call_args, _ = mock_action_head_forward_enabled.call_args
            combined_input = call_args[0]
            stag_context_vector = combined_input[:, self.agent_stag_enabled.hidden_dim:]
            # Assert that the context vector is not all zeros
            self.assertFalse(torch.all(stag_context_vector == 0))


        # --- Test STAG Disabled ---
        state = np.random.rand(self.obs_dim)
        _, _, _, activation_path, _ = self.agent_stag_disabled.perceive_and_update_state("test_cortex", state)

        with patch.object(self.agent_stag_disabled.action_head.layer, 'forward', return_value=torch.randn(1, self.action_dim)) as mock_action_head_forward_disabled:
            self.agent_stag_disabled.select_action(self.action_dim, activation_path)
            # The input to the action head's forward method is the combined_input
            call_args, _ = mock_action_head_forward_disabled.call_args
            combined_input = call_args[0]
            stag_context_vector = combined_input[:, self.agent_stag_disabled.hidden_dim:]
            # Assert that the context vector is all zeros
            self.assertTrue(torch.all(stag_context_vector == 0))
