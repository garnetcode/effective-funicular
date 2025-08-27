# Forcing re-compile to address caching issues.
import numpy as np
import os
import torch
import yaml
import logging
import random
from tqdm import tqdm
from django.core.management.base import BaseCommand
from api.services.chimera_agent import ChimeraAgent
from api.services.cortex.factory import create_cortex_configs_from_observation_space
import gymnasium as gym

logger = logging.getLogger(__name__)

def seed_all(s):
    """Sets the seed for all random number generators."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Command(BaseCommand):
    help = 'Train a ChimeraAgent using local Gymnasium environments.'

    def add_arguments(self, parser):
        parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to the main configuration file.")
        parser.add_argument("--env-curriculum", type=str, default="MountainCar-v0", help="A single env name or a comma-separated list of env names for curriculum learning.")
        parser.add_argument("--total-steps", type=int, default=50000, help="Total number of steps to train for across all environments.")
        parser.add_argument("--steps-per-env", type=int, default=50000, help="Number of steps to train on each environment in the curriculum.")

    def handle(self, *args, **options):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # --- Load Configuration ---
        config_path = os.path.join('', options['config'])
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at: {config_path}")
            return

        logger.info(f"Loaded configuration from {config_path}")

        agent_config = config.get('agent_config', {})
        training_config = config.get('training', {})
        hyperparams = agent_config.get('hyperparams', {})
        history_config = config.get('agent_history', {})

        # --- Set Seed for Reproducibility ---
        seed = training_config.get('seed')
        if seed is not None:
            logger.info(f"Setting random seed to {seed}")
            seed_all(seed)

        # --- Environment & Agent Setup ---
        env_names = [env.strip() for env in options['env_curriculum'].split(',')]
        total_training_steps = options['total_steps']
        steps_per_env = options['steps_per_env']

        # --- Pre-inspect all environments locally ---
        master_cortex_configs = {}
        max_action_dim = 0
        for name in env_names:
            logger.info(f"Inspecting local environment: {name}")
            try:
                temp_env = gym.make(name)
                env_cortex_configs, _ = create_cortex_configs_from_observation_space(temp_env.observation_space)
                master_cortex_configs.update(env_cortex_configs)
                if isinstance(temp_env.action_space, gym.spaces.Discrete):
                    max_action_dim = max(max_action_dim, temp_env.action_space.n)
                temp_env.close()
            except Exception as e:
                logger.error(f"Could not inspect environment {name}: {e}. Skipping.")
                env_names.remove(name)

        logger.info(f"Master cortex configuration will include: {list(master_cortex_configs.keys())}")
        logger.info(f"Max action dimension across all environments: {max_action_dim}")

        # --- Initialize a single, generalist agent ---
        agent_id = agent_config.get('default_agent_id_prefix', 'Kymera-') + "local-train"
        embedding_dim = agent_config.get('embedding_dim', 512)

        save_weights = training_config.get('save_weights', True)

        agent = ChimeraAgent(
            agent_id=agent_id,
            embedding_dim=embedding_dim,
            max_action_dim=max_action_dim,
            cortex_configs=master_cortex_configs,
            load_from_storage=save_weights,
            enable_saving=save_weights,
            hyperparams=hyperparams,
            history_config=history_config
        )
        logger.info(f"Initialized Generalist Agent '{agent_id}'")

        # --- Training Loop ---
        total_steps = 0
        episode_num = 0
        try:
            for env_id in env_names:
                logger.info(f"--- Starting Curriculum Stage: {env_id} ---")
                agent.set_active_skill(env_id)
                env = gym.make(env_id)
                _, cortex_id = create_cortex_configs_from_observation_space(env.observation_space)
                actual_action_dim = env.action_space.n

                steps_in_current_env = 0
                with tqdm(total=steps_per_env, desc=f"Training on {env_id}") as pbar:
                    while steps_in_current_env < steps_per_env and total_steps < total_training_steps:
                        state, _ = env.reset()
                        done, truncated = False, False
                        episode_reward = 0
                        episode_num += 1

                        while not (done or truncated):
                            h_t, z_t, h_normalized, activation_path, novelty = agent.perceive_and_update_state(cortex_id, state)
                            action, log_prob, _, _, _, _ = agent.select_action(actual_action_dim, activation_path)
                            next_state, reward, done, truncated, info = env.step(action)

                            agent.update_stag(h_normalized, reward)
                            agent.record_experience(h_t, z_t, activation_path, state, action, log_prob, reward, next_state, done or truncated)

                            state = next_state
                            episode_reward += reward
                            total_steps += 1
                            steps_in_current_env += 1
                            pbar.update(1)

                            # Online Training
                            policy_train_frequency = hyperparams.get('policy_train_frequency', 10)
                            if total_steps > hyperparams.get('burnin_steps', 1000) and \
                               total_steps % policy_train_frequency == 0 and \
                               len(agent.replay_buffer) > hyperparams.get('batch_size', 16):
                                train_stats = agent.train(cortex_id)
                                if train_stats:
                                    postfix_stats = {
                                        "WM Loss": f"{train_stats.get('wm_loss_total', 0):.4f}"
                                    }
                                    if 'ac_loss' in train_stats:
                                        postfix_stats["AC Loss"] = f"{train_stats['ac_loss']:.4f}"
                                    pbar.set_postfix(postfix_stats)

                        logger.info(f"Ep {episode_num} | Reward: {episode_reward:.2f} | Total Steps: {total_steps}")

                env.close()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        finally:
            if save_weights:
                agent.save_state(version_info={"message": "Local training stopped."})
                logger.info("Training finished and agent state saved.")
            else:
                logger.info("Training finished. Agent state not saved as per config.")
