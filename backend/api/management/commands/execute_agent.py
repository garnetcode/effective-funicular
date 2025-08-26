import numpy as np
import os
import time
import torch
import yaml
import logging
import asyncio
import pprint
import random
from tqdm import tqdm
from django.core.management.base import BaseCommand
from api.services.chimera_agent import ChimeraAgent
from colosseum_connector import ColosseumConnector
from api.services.cortex.factory import create_cortex_configs_from_observation_space
import gymnasium as gym

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Executes a trained ChimeraAgent in evaluation mode against a Colosseum server.'

    def add_arguments(self, parser):
        parser.add_argument("--config", type=str, required=True, help="Path to the main configuration file for the agent.")
        parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run.")

    def _deep_merge(self, source, destination):
        for key, value in source.items():
            if isinstance(value, dict):
                node = destination.setdefault(key, {})
                self._deep_merge(value, node)
            else:
                destination[key] = value
        return destination

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                main_config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at: {config_path}")
            return None

        if 'imports' in main_config:
            base_config = {}
            config_dir = os.path.dirname(config_path)
            for import_file in main_config['imports']:
                import_path = os.path.join(config_dir, import_file)
                imported_config = self._load_config(import_path)
                if imported_config:
                    base_config = self._deep_merge(base_config, imported_config)
            config = self._deep_merge(base_config, main_config)
        else:
            config = main_config
        return config

    def handle(self, *args, **options):
        asyncio.run(self.a_handle(*args, **options))

    async def a_handle(self, *args, **options):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        config_path = os.path.join('backend', options['config'])
        config = self._load_config(config_path)
        if not config:
            return

        env_name = config['env_name']
        agent_id = config.get('agent_id', f"agent-{env_name}")
        agent_config = config.get('agent_config', {})
        history_config = config.get('agent_history', {})
        num_episodes = options['num_episodes']

        connector = ColosseumConnector(env_name, agent_id)
        session_data = await connector.create_session()
        if not session_data:
            logger.error("Could not create Colosseum session. Exiting.")
            return

        try:
            obs_space_info = session_data['environment']['observation_space']
            observation_space = gym.spaces.Box(
                low=np.array(obs_space_info['low']), high=np.array(obs_space_info['high']),
                shape=obs_space_info['shape'], dtype=np.dtype(obs_space_info['dtype'])
            )
            actual_action_dim = session_data['environment']['action_space']['n']
            cortex_configs, cortex_id = create_cortex_configs_from_observation_space(observation_space)
        except (KeyError, TypeError) as e:
            logger.error(f"Could not determine environment specs from server response: {e}. Exiting.")
            return

        obs_dim = np.prod(observation_space.shape)
        if 'hyperparams' not in agent_config:
            agent_config['hyperparams'] = {}
        agent_config['hyperparams']['goal_dim'] = obs_dim

        agent = ChimeraAgent(
            agent_id=agent_id,
            embedding_dim=agent_config.get('embedding_dim', 256),
            max_action_dim=agent_config.get('max_action_dim', 256),
            latent_dim=agent_config.get('latent_dim', 64),
            hidden_dim=agent_config.get('hidden_dim', 128),
            cortex_configs=cortex_configs,
            load_from_storage=True, # Always load for execution
            hyperparams=agent_config.get('hyperparams', {}),
            history_config=history_config
        )
        agent.set_active_skill(env_name)
        logger.info(f"Agent '{agent.agent_id}' loaded. Starting execution for {num_episodes} episodes in '{env_name}'.")

        if not await connector.connect_websocket():
            logger.error("WebSocket connection failed. Exiting.")
            return

        join_response = await connector.join_session()
        if not join_response:
            logger.error("Failed to join session. Exiting.")
            await connector.close()
            return

        try:
            for i in range(num_episodes):
                logger.info(f"--- Starting Episode {i+1}/{num_episodes} ---")
                current_obs = np.array(session_data.get("observation"))
                episode_reward = 0
                done = False

                with tqdm(desc=f"Episode {i+1}") as pbar:
                    while not done:
                        _, _, _, activation_path, _ = agent.perceive_and_update_state(cortex_id, current_obs)
                        action, _, _, decision_maker, _, _ = agent.select_action(
                            actual_action_dim, activation_path, evaluation_mode=True
                        )

                        await connector.send_action(action)
                        msg = await connector.receive_message()

                        if not msg:
                            logger.warning("Disconnection detected. Ending episode.")
                            done = True
                            continue

                        if msg.get("type") == "action.taken":
                            current_obs = np.array(msg.get("observation"))
                            episode_reward += msg.get("reward")
                            done = msg.get("done")
                            pbar.update(1)
                            pbar.set_postfix({"Reward": f"{episode_reward:.2f}", "Decision": decision_maker})
                        elif msg.get("type") == "game.over":
                            done = True

                logger.info(f"--- Episode {i+1} Finished --- Total Reward: {episode_reward:.2f} ---")

                if i < num_episodes - 1:
                    logger.info("Resetting environment for next episode...")
                    reset_response = await connector.reset_environment()
                    if reset_response:
                        session_data['observation'] = reset_response.get("observation")
                    else:
                        logger.error("Failed to reset environment. Stopping.")
                        break
        finally:
            await connector.close()
            logger.info("Execution finished.")
