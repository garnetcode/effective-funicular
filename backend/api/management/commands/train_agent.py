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
import redis
import json
import time

logger = logging.getLogger(__name__)

# --- Redis-based UI State Caching ---
try:
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Successfully connected to Redis for UI state caching.")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Could not connect to Redis: {e}. UI updates will be disabled.")
    redis_client = None

def update_ui_state_in_redis(key, data):
    """Serializes data to JSON and stores it in a Redis key."""
    if redis_client is None:
        return
    try:
        # Using a custom JSON encoder to handle numpy types
        from api.services.chimera_agent import NumpyJSONEncoder
        payload = json.dumps(data, cls=NumpyJSONEncoder)
        redis_client.set(key, payload)
    except Exception as e:
        logger.warning(f"Failed to update UI state in Redis for key {key}: {e}")

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
        hyperparams = agent_config.get('hyperparams', {})
        history_config = config.get('agent_history', {})

        # --- Set Seed for Reproducibility ---
        seed = config.get('training', {}).get('seed')
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

        # Update UI state in Redis
        env_list_for_ui = [{'id': name, 'name': name} for name in env_names]
        update_ui_state_in_redis('chimera_environments', env_list_for_ui)

        # --- Initialize a single, generalist agent ---
        agent_id = agent_config.get('default_agent_id_prefix', 'Kymera-') + "local-train"
        embedding_dim = agent_config.get('embedding_dim', 512)

        agent = ChimeraAgent(
            agent_id=agent_id,
            embedding_dim=embedding_dim,
            max_action_dim=max_action_dim,
            cortex_configs=master_cortex_configs,
            load_from_storage=not config.get('force_new_agent', False),
            hyperparams=hyperparams,
            history_config=history_config
        )
        logger.info(f"Initialized Generalist Agent '{agent_id}'")

        # Update initial graph structure in Redis
        initial_graph = agent.get_graph_structure()
        update_ui_state_in_redis('chimera_graph_state', initial_graph)


        # --- Training Loop ---
        total_steps = 0
        episode_num = 0
        try:
            for env_id in env_names:
                logger.info(f"--- Starting Curriculum Stage: {env_id} ---")
                env = gym.make(env_id)
                _, cortex_id = create_cortex_configs_from_observation_space(env.observation_space)
                actual_action_dim = env.action_space.n
                agent.set_active_skill(env_id, actual_action_dim)

                steps_in_current_env = 0
                # Initialize the postfix dictionary for the progress bar
                postfix_data = {
                    "Ep": 0, "Reward": "N/A", "Epsilon": "N/A", "Decision": "N/A",
                    "WM Loss": "N/A", "AC Loss": "N/A", "Nodes": 0, "Edges": 0
                }
                with tqdm(total=steps_per_env, desc=f"Training on {env_id}") as pbar:
                    while steps_in_current_env < steps_per_env and total_steps < total_training_steps:
                        state, _ = env.reset()
                        done, truncated = False, False
                        episode_reward = 0
                        episode_num += 1
                        postfix_data["Ep"] = episode_num

                        while not (done or truncated):
                            h_t, z_t, h_normalized, activation_path, novelty = agent.perceive_and_update_state(cortex_id, state)
                            action, log_prob, _, decision_maker, epsilon, _, action_probs = agent.select_action(actual_action_dim, activation_path)

                            # Update postfix data for real-time display
                            postfix_data["Epsilon"] = f"{epsilon:.2f}"
                            postfix_data["Decision"] = decision_maker
                            pbar.set_postfix(postfix_data)

                            # Update action probabilities in Redis for UI visualization
                            update_ui_state_in_redis('chimera_action_update', {'probabilities': action_probs.tolist()})

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
                                    # Update postfix with training stats, checking for type before formatting
                                    wm_loss = train_stats.get('wm_loss_total')
                                    ac_loss = train_stats.get('ac_loss')
                                    postfix_data["WM Loss"] = f"{wm_loss:.4f}" if isinstance(wm_loss, (int, float)) else "N/A"
                                    postfix_data["AC Loss"] = f"{ac_loss:.4f}" if isinstance(ac_loss, (int, float)) else "N/A"

                                    # Update UI State in Redis after training step
                                    update_ui_state_in_redis('chimera_training_metrics', train_stats)
                                    updated_graph = agent.get_graph_structure()
                                    update_ui_state_in_redis('chimera_graph_state', updated_graph)

                                    # Update graph size in postfix
                                    postfix_data["Nodes"] = len(updated_graph.get('nodes', []))
                                    postfix_data["Edges"] = len(updated_graph.get('edges', []))


                        # End of episode
                        postfix_data["Reward"] = f"{episode_reward:.2f}"
                        pbar.set_postfix(postfix_data)

                        # Also update episode stats in Redis
                        episode_stats = {
                            'episode': episode_num,
                            'reward': episode_reward,
                            'total_steps': total_steps
                        }
                        update_ui_state_in_redis('chimera_episode_metrics', episode_stats)

                env.close()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        finally:
            agent.save_state(version_info={"message": "Local training stopped."})
            logger.info("Training finished and agent state saved.")
