import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

# This is a bit of a hack to make the script importable
import sys
sys.path.append('backend')
import run_colosseum_agents
import gymnasium as gym
import numpy as np
import torch

class RunColosseumAgentsTests(unittest.TestCase):

    @patch('run_colosseum_agents.yaml.safe_load')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('run_colosseum_agents.ChimeraAgent')
    @patch('run_colosseum_agents.ColosseumConnector')
    @patch('run_colosseum_agents.gym.make')
    def test_curriculum_workflow(self, mock_gym_make, mock_connector, mock_chimera_agent, mock_open, mock_safe_load):
        """
        Tests that the main script iterates through the curriculum and calls the agent correctly.
        """
        # --- Setup Mocks ---
        # Mock the config file loading
        mock_safe_load.return_value = {
            'episodes_per_env': 1,
            'agent_config': {'embedding_dim': 128, 'hyperparams': {}},
            'agent_history': {}
        }

        # Mock the gym environment inspection
        mock_env_instance = MagicMock()
        mock_env_instance.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        mock_env_instance.action_space = gym.spaces.Discrete(2)
        mock_gym_make.return_value = mock_env_instance

        # Mock the ChimeraAgent
        mock_agent_instance = MagicMock()
        mock_agent_instance.perceive_and_update_state.return_value = (
            torch.zeros(1, 512),
            torch.zeros(1, 128),
            np.zeros(512),
            [],
            0.0,
            None  # winner_id
        )
        mock_agent_instance.select_action.return_value = (
            0,
            torch.tensor(0.1),
            torch.zeros(1, 128),
            "policy",
            0.1
        )
        mock_agent_instance.train.return_value = {'policy_loss': 0.1234}
        mock_chimera_agent.return_value = mock_agent_instance

        # Mock the ColosseumConnector
        mock_connector_instance = MagicMock()
        # The create_session response needs to be realistic enough for the factory
        mock_obs_space_info = {
            'type': 'Box', 'shape': (4,), 'low': [-1.0] * 4, 'high': [1.0] * 4, 'dtype': 'float32'
        }
        mock_action_space_info = {'type': 'discrete', 'n': 2}
        session_data = {
            'success': True,
            'observation': [0.1, 0.2, 0.3, 0.4],
            'environment': {
                'observation_space': mock_obs_space_info,
                'action_space': mock_action_space_info
            }
        }
        mock_connector_instance.create_session = AsyncMock(return_value=session_data)
        mock_connector_instance.connect_websocket = AsyncMock(return_value=True)
        mock_connector_instance.join_session = AsyncMock(return_value={'type': 'agent.joined'})
        mock_connector_instance.receive_message = AsyncMock(return_value={'type': 'game.over'}) # End episode immediately
        mock_connector_instance.send_action = AsyncMock()
        mock_connector_instance.close = AsyncMock() # Make close awaitable
        mock_connector.return_value = mock_connector_instance

        # --- Run the main function ---
        # We need to run the async main function of the script
        asyncio.run(run_colosseum_agents.run_training_curriculum())

        # --- Assertions ---
        # 1. Assert that ChimeraAgent was initialized only once
        mock_chimera_agent.assert_called_once()

        # 2. Assert that set_active_skill was called for each environment in the curriculum
        # The script's default curriculum is ["CartPole-v1", "LunarLander-v2"]
        self.assertEqual(mock_agent_instance.set_active_skill.call_count, 2)
        call_args_list = mock_agent_instance.set_active_skill.call_args_list
        self.assertEqual(call_args_list[0].args[0], "CartPole-v1")
        self.assertEqual(call_args_list[1].args[0], "LunarLander-v2")

        # 3. Assert that a connector was created for each environment
        # It's called once for spec inspection and once for training, for each of the 2 envs.
        self.assertEqual(mock_connector.call_count, 4)

if __name__ == '__main__':
    unittest.main()
