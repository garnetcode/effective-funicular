import numpy as np
import os
from django.core.management.base import BaseCommand
from api.services.chimera_agent import ChimeraAgent

try:
    import gymnasium as gym
except ImportError:
    # Set gym to None if it's not installed, so the command can still be loaded by Django
    gym = None

def get_env_config(env):
    """Inspects a gymnasium environment to determine agent configuration."""
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

import yaml
import logging

# --- Set up logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Train a ChimeraAgent in a Gymnasium environment using a config file.'

    def add_arguments(self, parser):
        parser.add_argument("--config", type=str, default="backend/config.yaml", help="Path to the configuration file.")

    def handle(self, *args, **options):
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
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # --- Agent Initialization ---
        cortex_configs, cortex_id, _ = get_env_config(env)
        agent_id = config.get('agent_id', f"agent-{env_name}")

        agent = ChimeraAgent(
            agent_id=agent_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=config.get('latent_dim', 64),
            hidden_dim=config.get('hidden_dim', 128),
            cortex_configs=cortex_configs,
            load_from_storage=not config.get('force_new_agent', False),
            hyperparams={
                'learning_rate': config.get('learning_rate', 0.005),
                'gamma': config.get('gamma', 0.99)
            }
        )

        logger.info(f"Starting training for agent '{agent_id}' in '{env_name}'...")
        logger.info(f"Agent dimensions: obs={obs_dim}, action={action_dim}, latent={agent.latent_dim}, hidden={agent.hidden_dim}")

        total_rewards = []
        for episode in range(config.get('episodes', 1000)):
            state, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0

            # Reset agent's internal state at the start of each episode
            agent.hidden_state = torch.zeros(1, agent.hidden_dim)
            agent.last_action = torch.tensor(0)

            while not (terminated or truncated):
                # --- Homeostasis Integration ---
                old_energy = agent.energy
                old_integrity = agent.integrity

                # 2. Perceive state and select action
                agent.perceive_and_update_state(cortex_id, state)
                action, log_prob = agent.select_action()

                # 3. Step the environment
                next_state, external_reward, terminated, truncated, info = env.step(action)

                # 4. Update agent vitals
                agent.energy -= agent.metabolic_cost
                agent.energy += info.get('energy_change', 0.0)
                agent.integrity += info.get('integrity_change', 0.0)
                agent.energy = min(agent.energy, agent.max_energy)

                # 5. Calculate homeostatic reward
                energy_reward = agent.energy - old_energy
                integrity_reward = agent.integrity - old_integrity
                homeostatic_reward = energy_reward + integrity_reward

                # 6. Calculate total reward
                total_reward = external_reward + homeostatic_reward

                # 7. Record experience with the total reward
                agent.record_experience(state, action, log_prob, total_reward, next_state, (terminated or truncated))

                state = next_state
                episode_reward += external_reward

            train_stats = agent.train()
            agent.save_state(version_info=train_stats) # Save a new version after training
            total_rewards.append(episode_reward)

            if episode % 10 == 0:
                avg_reward = np.mean(total_rewards[-100:])
                logger.info(
                    f"Episode {episode}/{config.get('episodes', 1000)} | "
                    f"Total Reward: {episode_reward:.2f} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"WM Loss: {train_stats.get('world_model_loss', 'N/A'):.4f} | "
                    f"Policy Loss: {train_stats.get('policy_loss', 'N/A'):.4f}"
                )

        env.close()
        logger.info("Training finished.")
        logger.info(f"Final agent state saved to {agent.storage_path}")
