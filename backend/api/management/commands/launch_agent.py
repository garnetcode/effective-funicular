import logging
import yaml
import numpy as np
import time
import sys
import os
import torch
import asyncio
from collections import namedtuple

from django.core.management.base import BaseCommand
from api.services.chimera_agent import ChimeraAgent
from api.services.redis_buffer import RedisBuffer
from api.services.experience import Experience
from colosseum_connector import ColosseumConnector
from api.services.redis_ui_utils import update_ui_state_in_redis
from api.services.cortex.factory import create_cortex_configs_from_observation_space
import gymnasium as gym
import redis
import pickle

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

class Command(BaseCommand):
    help = 'Runs the Actor process for a ChimeraAgent, which collects experience.'

    def add_arguments(self, parser):
        parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to the main configuration file.")
        parser.add_argument("--env-id", type=str, default="LunarLander-v3", help="The Colosseum environment ID to run in.")
        parser.add_argument("--agent-tag", type=str, default=None, help="A unique tag for the agent.")
        parser.add_argument("--port", type=int, default=8002, help="The port of the Colosseum server.")

    def handle(self, *args, **options):
        env_id = options['env_id']
        agent_tag = options['agent_tag'] or f"chimera-agent-{env_id}"
        config_path = options['config']
        port = options['port']

        logger = setup_logging('actor', env_id)

        try:
            asyncio.run(self.run_actor(logger, env_id, agent_tag, config_path, port))
        except KeyboardInterrupt:
            logger.info("\nActor process interrupted by user.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in actor: {e}", exc_info=True)

    async def run_actor(self, logger, env_id, agent_tag, config_path, port):
        logger.info(f"Starting actor for agent '{agent_tag}' on env '{env_id}'")

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at: {config_path}")
            return

        hyperparams = config.get('agent_config', {}).get('hyperparams', {})
        hyperparams['use_planner'] = True

        connector = ColosseumConnector(env_id, agent_tag, http_port=port, ws_port=port)
        session_info = await connector.create_session()
        if not session_info:
            logger.error("Could not create Colosseum session. Exiting.")
            return

        env_specs = session_info['environment']
        obs_space_info = env_specs['observation_space']
        action_space_info = env_specs['action_space']

        cortex_configs, cortex_id = create_cortex_configs_from_observation_space(
            gym.spaces.Box(
                low=np.array(obs_space_info['low']), high=np.array(obs_space_info['high']),
                shape=obs_space_info['shape'], dtype=np.dtype(obs_space_info['dtype'])
            )
        )

        redis_buffer = RedisBuffer()

        actor_agent = ChimeraAgent(
            agent_id=agent_tag,
            embedding_dim=config.get('agent_config', {}).get('embedding_dim', 512),
            max_action_dim=action_space_info['n'],
            cortex_configs=cortex_configs,
            hyperparams=hyperparams,
            replay_buffer=None, # The actor does not use the buffer for training
            load_from_storage=False, # We will load weights manually
            history_config={}
        )
        actor_agent.set_active_skill(env_id)

        try:
            redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
            redis_client.ping()
            logger.info("Actor connected to Redis for state updates.")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Actor could not connect to Redis: {e}. Will not receive model updates.")
            redis_client = None

        last_known_version = -1

        if not await connector.connect_websocket() or not await connector.join_session():
            logger.error("Could not connect to session. Exiting.")
            await connector.close()
            return

        current_obs = np.array(session_info.get("observation"))
        episode_count = 0

        try:
            while True:
                episode_count += 1
                logger.info(f"--- Starting Episode {episode_count} ---")

                # Check for new weights from the learner
                if redis_client:
                    try:
                        current_version = int(redis_client.get('chimera_agent_state:version') or -1)
                        if current_version > last_known_version:
                            logger.info(f"New agent version {current_version} found. Loading from Redis.")
                            serialized_state = redis_client.get('chimera_agent_state:latest')
                            if serialized_state:
                                state_dict = pickle.loads(serialized_state)
                                actor_agent.load_actor_state_dict(state_dict)
                                last_known_version = current_version
                    except Exception as e:
                        logger.error(f"Error loading agent state from Redis: {e}")


                done = False
                episode_reward = 0

                while not done:
                    h_t, z_t, h_normalized, activation_path, novelty, winner_id = actor_agent.perceive_and_update_state(cortex_id, current_obs)
                    action, log_prob, stag_ctx, decision_maker, epsilon, _, action_probs, h_norm = actor_agent.select_action(
                        action_space_info['n'], activation_path, evaluation_mode=False
                    )
                    if decision_maker == 'policy':
                        log_msg = (
                            f"Step {actor_agent.steps_done}: Action: {action}, Mode: {decision_maker}, Epsilon: {epsilon:.4f}, "
                            f"h_norm: {h_norm.norm():.4f}, stag_ctx_norm: {stag_ctx.norm():.4f}"
                        )
                        logger.info(log_msg)
                    else:
                        logger.info(f"Step {actor_agent.steps_done}: Action: {action}, Mode: {decision_maker}, Epsilon: {epsilon:.4f}")

                    # Update UI with actor state
                    actor_state_for_ui = {
                        'h_t': h_t[0].detach().cpu().numpy().flatten(), # Top-level hidden state
                        'z_t': z_t[0].detach().cpu().numpy().flatten(), # Top-level latent state
                        'epsilon': epsilon,
                        'action_probs': action_probs
                    }
                    update_ui_state_in_redis('chimera_actor_state', actor_state_for_ui)


                    await connector.send_action(int(action))
                    msg = await connector.receive_message()
                    if not msg:
                        logger.warning("Actor did not receive state from server. Ending episode.")
                        break

                    if msg.get("type") == "action.taken":
                        next_obs = np.array(msg.get("observation"))
                        reward = msg.get("reward", 0)
                        done = msg.get("done", False)

                        experience = Experience(h_t, z_t, activation_path, current_obs, action, log_prob, reward, next_obs, done, actor_agent.current_goal, winner_id)
                        redis_buffer.push(experience)

                        if h_normalized is not None:
                            actor_agent.update_stag(h_normalized, reward)

                        episode_reward += reward
                        current_obs = next_obs
                    elif msg.get("type") == "game.over":
                        next_obs = np.array(msg.get("observation", current_obs))
                        reward = msg.get("reward", 0)

                        experience = Experience(h_t, z_t, activation_path, current_obs, action, log_prob, reward, next_obs, True, actor_agent.current_goal, winner_id)
                        redis_buffer.push(experience)

                        episode_reward += reward
                        done = True

                logger.info(f"Episode {episode_count} finished. Reward: {episode_reward:.2f}")

                reset_response = await connector.reset_environment()
                if reset_response:
                    current_obs = np.array(reset_response.get("observation"))
                else:
                    logger.error("Failed to reset environment. Stopping actor.")
                    break
        finally:
            await connector.close()
            logger.info("Actor process finished.")
