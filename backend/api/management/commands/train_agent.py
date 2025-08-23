import numpy as np
import os
import torch
import yaml
import logging
from tqdm import tqdm
from django.core.management.base import BaseCommand
from api.services.chimera_agent import ChimeraAgent

try:
    import gymnasium as gym
except ImportError:
    gym = None

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Train a ChimeraAgent in a Gymnasium environment using a config file.'

    def add_arguments(self, parser):
        parser.add_argument("--config", type=str, default="backend/config.yaml", help="Path to the configuration file.")

    @staticmethod
    def get_env_config(env):
        """Inspects a gymnasium environment to determine agent configuration."""
        if not gym:
            raise ImportError("Gymnasium library not found.")

        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Box) or len(obs_space.shape) != 1:
            raise NotImplementedError("Only 1D Box observation spaces are supported.")

        act_space = env.action_space
        if not isinstance(act_space, gym.spaces.Discrete):
            raise NotImplementedError("Only Discrete action spaces are supported.")

        input_dim = obs_space.shape[0]
        cortex_configs = {"vector_input": {"type": "DenseCortex", "params": {"input_dim": input_dim}}}
        n_actions = act_space.n
        return cortex_configs, "vector_input", n_actions

    def _safe_format(self, value, format_spec):
        """Safely format a value, returning a placeholder if it's not a number."""
        if isinstance(value, (int, float)):
            return f"{value:{format_spec}}"
        return str(value)

    def handle(self, *args, **options):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("training.log"),
                logging.StreamHandler()
            ]
        )

        if not gym:
            logger.error("FATAL: gymnasium library not found. Please install it with `pip install gymnasium`")
            return

        # --- Load Configuration ---
        config_path = options['config']
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at: {config_path}")
            return

        logger.info(f"Loaded configuration from {config_path}")

        # --- Environment Setup ---
        env_name = config['env_name']
        logger.info(f"Initializing environment: {env_name}")
        env = gym.make(env_name)

        # --- Agent Initialization ---
        cortex_configs, cortex_id, action_dim = self.get_env_config(env)
        obs_dim = env.observation_space.shape[0]
        agent_id = config.get('agent_id', f"agent-{env_name}")

        agent_config = config.get('agent_config', {})
        agent = ChimeraAgent(
            agent_id=agent_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=agent_config.get('latent_dim', 64),
            hidden_dim=agent_config.get('hidden_dim', 128),
            cortex_configs=cortex_configs,
            load_from_storage=not config.get('force_new_agent', False),
            hyperparams=agent_config.get('hyperparams', {})
        )

        logger.info(f"Starting training for agent '{agent.agent_id}' in '{env_name}'...")

        total_rewards = []
        num_episodes = config.get('episodes_per_env', 100)

        # --- Training Loop with tqdm ---
        with tqdm(total=num_episodes, desc="Training Agent") as pbar:
            for episode in range(num_episodes):
                state, info = env.reset()
                terminated = False
                truncated = False
                episode_reward = 0

                agent.hidden_state = torch.zeros(1, agent.hidden_dim)
                agent.last_action = torch.tensor(0)

                while not (terminated or truncated):
                    agent.perceive_and_update_state(cortex_id, state)
                    action, log_prob = agent.select_action()
                    next_state, external_reward, terminated, truncated, info = env.step(action)

                    agent.energy -= agent.metabolic_cost
                    agent.energy = min(agent.energy, agent.max_energy)

                    total_reward = external_reward  # Simplified reward for this env
                    agent.record_experience(state, action, log_prob, total_reward, next_state, (terminated or truncated))

                    state = next_state
                    episode_reward += external_reward

                train_stats = agent.train()
                agent.save_state(version_info=train_stats)
                total_rewards.append(episode_reward)

                # Update tqdm progress bar with useful stats
                avg_reward = np.mean(total_rewards[-100:])
                pbar.set_postfix({
                    "Reward": f"{episode_reward:.2f}",
                    "Avg Reward": f"{avg_reward:.2f}",
                    "WM Loss": self._safe_format(train_stats.get('world_model_loss'), '.4f'),
                    "Policy Loss": self._safe_format(train_stats.get('policy_loss'), '.4f')
                })
                pbar.update(1)

        env.close()
        logger.info("Training finished.")
        logger.info(f"Final agent state saved to {agent.history_manager.storage_dir}")
