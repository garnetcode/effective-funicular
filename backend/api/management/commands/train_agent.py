import numpy as np
import os
import torch
import yaml
import logging
import asyncio
from tqdm import tqdm
from django.core.management.base import BaseCommand
from api.services.chimera_agent import ChimeraAgent
from colosseum_connector import ColosseumConnector

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Train a ChimeraAgent against a Colosseum server.'

    def add_arguments(self, parser):
        parser.add_argument("--config", type=str, default="backend/config.yaml", help="Path to the configuration file.")

    def _safe_format(self, value, format_spec):
        """Safely format a value, returning a placeholder if it's not a number."""
        if isinstance(value, (int, float)):
            return f"{value:{format_spec}}"
        return str(value)

    def handle(self, *args, **options):
        # We are calling an async method from a sync command.
        asyncio.run(self.a_handle(*args, **options))

    async def a_handle(self, *args, **options):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("training.log"),
                logging.StreamHandler()
            ]
        )

        # --- Load Configuration ---
        config_path = options['config']
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at: {config_path}")
            return

        logger.info(f"Loaded configuration from {config_path}")

        # --- Environment & Agent Setup ---
        env_name = config['env_name']
        agent_id = config.get('agent_id', f"agent-{env_name}")
        agent_config = config.get('agent_config', {})
        history_config = config.get('agent_history', {})
        num_episodes = config.get('episodes_per_env', 100)

        # HACK: Dimensions should ideally come from server. Using CartPole-v1 defaults.
        obs_dim = 4
        action_dim = 2

        # --- Create a single, persistent agent ---
        agent = ChimeraAgent(
            agent_id=agent_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=agent_config.get('latent_dim', 64),
            hidden_dim=agent_config.get('hidden_dim', 128),
            cortex_configs={"vector_input": {"type": "DenseCortex", "params": {"input_dim": obs_dim}}},
            load_from_storage=not config.get('force_new_agent', False),
            hyperparams=agent_config.get('hyperparams', {}),
            history_config=history_config
        )
        logger.info(f"Agent '{agent.agent_id}' created. Starting Colosseum training for {num_episodes} episodes in '{env_name}'...")

        total_rewards = []

        with tqdm(total=num_episodes, desc="Training on Colosseum") as pbar:
            for episode in range(num_episodes):
                # --- Create a new session for each episode ---
                connector = ColosseumConnector(env_name, agent_id)
                join_response = await connector.connect()
                if not join_response:
                    logger.error(f"Episode {episode + 1}: Failed to join session. Skipping.")
                    await connector.close()
                    pbar.update(1)
                    continue

                # --- Gameplay Loop ---
                current_obs = np.array(join_response.get("observation"))
                episode_reward = 0
                done = False

                while not done:
                    agent.perceive_and_update_state("vector_input", current_obs)
                    action, log_prob, stag_context = agent.select_action()

                    await connector.send_action(action)
                    msg = await connector.receive_message()

                    if not msg or msg.get("type") != "action.taken":
                        logger.warning(f"Unexpected message or disconnection: {msg}")
                        break

                    next_obs = np.array(msg.get("observation"))
                    reward = msg.get("reward")
                    done = msg.get("done")

                    total_reward = reward  # Simplified reward
                    agent.record_experience(agent.hidden_state, stag_context, current_obs, action, log_prob, total_reward, next_obs, done)

                    current_obs = next_obs
                    episode_reward += reward

                # --- Post-Episode ---
                await connector.close()
                train_stats = agent.train()
                agent.save_state(version_info=train_stats)
                total_rewards.append(episode_reward)

                avg_reward = np.mean(total_rewards[-100:])
                pbar.set_postfix({
                    "Reward": f"{episode_reward:.2f}",
                    "Avg Rwd": f"{avg_reward:.2f}",
                    "Policy Loss": self._safe_format(train_stats.get('policy_loss'), '.4f')
                })
                pbar.update(1)

        logger.info("Training finished.")
        logger.info(f"Final agent state saved to {agent.history_manager.storage_dir}")
