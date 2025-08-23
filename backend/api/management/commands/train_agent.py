import numpy as np
import os
import torch
import yaml
import logging
from django.core.management.base import BaseCommand
from api.services.chimera_agent import ChimeraAgent

try:
    import gymnasium as gym
except ImportError:
    gym = None

# --- Set up logging ---
# Moved to a more standard location at the top
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
        obs_dim = env.observation_space.shape[0] # obs_dim is the input to the cortex
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
        logger.info(f"Agent dimensions: obs={agent.obs_dim}, action={agent.action_dim}, latent={agent.latent_dim}, hidden={agent.hidden_dim}")

        total_rewards = []
        num_episodes = config.get('episodes_per_env', 100) # Use the correct config key

        for episode in range(num_episodes):
            state, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0

            # Reset agent's internal state at the start of each episode
            agent.hidden_state = torch.zeros(1, agent.hidden_dim)
            agent.last_action = torch.tensor(0)

            while not (terminated or truncated):
                # Perceive state and select action
                agent.perceive_and_update_state(cortex_id, state)
                action, log_prob = agent.select_action()

                # Step the environment
                next_state, external_reward, terminated, truncated, info = env.step(action)

                # Update agent vitals (homeostasis)
                agent.energy -= agent.metabolic_cost
                agent.energy = min(agent.energy, agent.max_energy)

                # For this simple env, we can use a placeholder for homeostatic reward
                homeostatic_reward = 0.0
                total_reward = external_reward + homeostatic_reward

                # Record experience with the total reward
                agent.record_experience(state, action, log_prob, total_reward, next_state, (terminated or truncated))

                state = next_state
                episode_reward += external_reward

            # Post-episode training and state saving
            train_stats = agent.train()
            agent.save_state(version_info=train_stats)
            total_rewards.append(episode_reward)

            if episode % 10 == 0 or episode == num_episodes - 1:
                avg_reward = np.mean(total_rewards[-100:])
                wm_loss = self._safe_format(train_stats.get('world_model_loss'), '.4f')
                policy_loss = self._safe_format(train_stats.get('policy_loss'), '.4f')

                log_message = (
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Avg Reward (last 100): {avg_reward:.2f} | "
                    f"WM Loss: {wm_loss} | "
                    f"Policy Loss: {policy_loss}"
                )
                logger.info(log_message)

        env.close()
        logger.info("Training finished.")
        # Fix: Use the correct attribute to get the storage path
        logger.info(f"Final agent state saved to {agent.history_manager.storage_dir}")
