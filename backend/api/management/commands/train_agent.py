import logging
import yaml
import numpy as np
import time
import sys
import os
import torch
from collections import namedtuple

from django.core.management.base import BaseCommand
from api.services.chimera_agent import ChimeraAgent, NumpyJSONEncoder
from api.services.redis_buffer import RedisBuffer
from api.services.cortex.factory import create_cortex_configs_from_observation_space
import gymnasium as gym
import redis
import json

def setup_logging(process_name, environment_name):
    logger = logging.getLogger(f"chimera.{process_name}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_filename = f"{environment_name.replace('/', '_')}_{process_name}.log"
    file_handler = logging.FileHandler(log_filename)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

try:
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
except redis.exceptions.ConnectionError as e:
    logging.error(f"Could not connect to Redis for UI: {e}.")
    redis_client = None

def update_ui_state_in_redis(key, data):
    if redis_client is None: return
    try:
        payload = json.dumps(data, cls=NumpyJSONEncoder)
        redis_client.set(key, payload)
    except Exception as e:
        logging.warning(f"Failed to update UI state in Redis for key {key}: {e}")

class Command(BaseCommand):
    help = 'Runs the Learner process for a ChimeraAgent.'

    def add_arguments(self, parser):
        parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to the main configuration file.")
        parser.add_argument("--env-id", type=str, default="LunarLander-v3", help="A name for the environment, used for loading agent history.")
        parser.add_argument("--agent-tag", type=str, default=None, help="A unique tag for the agent.")

    def handle(self, *args, **options):
        env_id = options['env_id']
        agent_tag = options['agent_tag'] or f"chimera-agent-{env_id}"
        config_path = options['config']

        logger = setup_logging('learner', env_id)
        logger.info(f"Starting learner for agent '{agent_tag}' on env '{env_id}'")

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at: {config_path}")
            return

        hyperparams = config.get('agent_config', {}).get('hyperparams', {})

        try:
            dummy_env = gym.make(env_id)
            cortex_configs, cortex_id = create_cortex_configs_from_observation_space(dummy_env.observation_space)
            max_action_dim = dummy_env.action_space.n
            dummy_env.close()
        except Exception as e:
            logger.error(f"Could not create dummy environment '{env_id}' to get specs: {e}")
            return

        redis_buffer = RedisBuffer()

        learner_agent = ChimeraAgent(
            agent_id=agent_tag,
            embedding_dim=config.get('agent_config', {}).get('embedding_dim', 512),
            max_action_dim=max_action_dim,
            cortex_configs=cortex_configs,
            hyperparams=hyperparams,
            replay_buffer=None,
            load_from_storage=not config.get('force_new_agent', False),
            history_config=config.get('agent_history', {})
        )
        learner_agent.set_active_skill(env_id)

        logger.info("Learner setup complete. Starting training loop...")

        train_steps = 0
        weights_save_path = "latest_agent.pt"

        try:
            while True:
                batch_size = hyperparams.get('batch_size', 50)
                if len(redis_buffer) > hyperparams.get('burnin_steps', 1000):

                    batch_list = redis_buffer.sample(batch_size)
                    if not batch_list:
                        time.sleep(1)
                        continue

                    # The experience is a tuple. Let's define a namedtuple for clarity.
                    Experience = namedtuple('Experience', ['h_t', 'z_t', 'activation_path', 'obs', 'action', 'log_prob', 'reward', 'next_obs', 'done', 'goal'])

                    # Convert list of serialized tuples to a batch dictionary
                    experiences = [Experience(*exp) for exp in batch_list]
                    batch = {field: np.array([getattr(e, field) for e in experiences]) for field in Experience._fields}

                    train_stats = learner_agent.train_on_batch(batch, cortex_id=cortex_id)
                    train_steps += 1

                    if train_stats:
                        update_ui_state_in_redis('chimera_training_metrics', train_stats)

                        if train_steps % 20 == 0:
                            stats_str = ", ".join([f"{key}={value:.4f}" for key, value in train_stats.items() if isinstance(value, (int, float))])
                            logger.info(f"Train step {train_steps}: {stats_str}")

                    if train_steps % hyperparams.get('actor_update_frequency', 100) == 0:
                        logger.info(f"Saving latest weights to {weights_save_path}")
                        torch.save(learner_agent.state_dict(), weights_save_path)

                    if train_steps % hyperparams.get('stag_ui_update_frequency', 50) == 0:
                        stag_graph = learner_agent.get_graph_structure(env_id)
                        if stag_graph:
                            update_ui_state_in_redis('chimera_stag_graph', stag_graph)
                else:
                    logger.info(f"Waiting for replay buffer to fill ({len(redis_buffer)}/{hyperparams.get('burnin_steps', 1000)})...")
                    time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Shutting down.")
        finally:
            logger.info("Finalizing and saving model before exit.")
            torch.save(learner_agent.state_dict(), weights_save_path)
