import asyncio
import logging
import yaml
import numpy as np
import torch
import websockets
import random
from tqdm import tqdm

from django.core.management.base import BaseCommand
from api.services.chimera_agent import ChimeraAgent
from colosseum_connector import ColosseumConnector
from api.services.cortex.factory import create_cortex_configs_from_observation_space
import gymnasium as gym
import redis
import json
import time

# --- Setup logging ---
logger = logging.getLogger(__name__)

# --- Redis Caching for UI ---
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
        payload = json.dumps(data, cls=NumpyJSONEncoder)
        redis_client.set(key, payload)
    except Exception as e:
        logger.warning(f"Failed to update UI state in Redis for key {key}: {e}")

class NumpyJSONEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Command(BaseCommand):
    help = 'Train a ChimeraAgent using a Colosseum server environment.'

    def add_arguments(self, parser):
        parser.add_argument("--config", type=str, default="backend/configs/base.yaml", help="Path to the main configuration file.")
        parser.add_argument("--env-id", type=str, default="LunarLander-v2", help="The Colosseum environment ID to train in.")
        parser.add_argument("--episodes", type=int, default=500, help="Total number of episodes to train for.")
        parser.add_argument("--agent-tag", type=str, default=None, help="A unique tag for the agent.")
        parser.add_argument("--port", type=int, default=8002, help="The port of the Colosseum server.")

    def handle(self, *args, **options):
        """Synchronous entry point that runs the async handler."""
        try:
            asyncio.run(self.async_handle(*args, **options))
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in async_handle: {e}", exc_info=True)

    async def async_handle(self, *args, **options):
        """The main asynchronous logic for training the agent."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        env_id = options['env_id']
        agent_tag = options['agent_tag'] or f"chimera-agent-{random.randint(1000, 9999)}"
        config_path = options['config']
        episodes = options['episodes']
        port = options['port']

        logger.info(f"Starting Chimera agent '{agent_tag}' for environment: {env_id} on port {port}")

        # --- Load Configuration ---
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except FileNotFoundError:
            logger.error(f"Config file not found at: {config_path}")
            return

        agent_config = config.get('agent_config', {})
        hyperparams = agent_config.get('hyperparams', {})
        history_config = config.get('agent_history', {})

        # --- Colosseum Connection ---
        connector = ColosseumConnector(env_id, agent_tag, port=port)

        try:
            session_info = await connector.create_session()
            if not session_info:
                logger.error("Could not create session. Exiting.")
                return

            if not await connector.connect_websocket():
                logger.error("Could not connect to WebSocket. Exiting.")
                return

            join_response = await connector.join_session()
            if not join_response:
                logger.error("Could not join session. Exiting.")
                await connector.close()
                return

            # --- Environment and Agent Setup (Post-Connection) ---
            env_specs = join_response['environment']
            obs_space_info = env_specs['observation_space']
            action_space_info = env_specs['action_space']

            observation_space = gym.spaces.Box(
                low=np.array(obs_space_info['low']),
                high=np.array(obs_space_info['high']),
                shape=obs_space_info['shape'],
                dtype=np.dtype(obs_space_info['dtype'])
            )
            cortex_configs, cortex_id = create_cortex_configs_from_observation_space(observation_space)
            actual_action_dim = action_space_info['n']

            agent = ChimeraAgent(
                agent_id=agent_tag,
                embedding_dim=agent_config.get('embedding_dim', 512),
                max_action_dim=actual_action_dim,
                cortex_configs=cortex_configs,
                load_from_storage=not config.get('force_new_agent', False),
                hyperparams=hyperparams,
                history_config=history_config
            )
            agent.set_active_skill(env_id, actual_action_dim)
            logger.info(f"Initialized Chimera Agent '{agent_tag}' for {env_id}")

            # --- Main Training Loop ---
            postfix_data = {
                "Reward": "N/A", "Epsilon": "N/A", "Decision": "N/A",
                "WM Loss": "N/A", "AC Loss": "N/A", "Nodes": 0, "Edges": 0
            }
            with tqdm(total=episodes, desc=f"Training on {env_id}") as pbar:
                for episode in range(1, episodes + 1):
                    current_obs = np.array(join_response.get("observation"))
                    done = False
                    episode_reward = 0
                    pbar.set_postfix(postfix_data)

                    while not done:
                        h_t, z_t, h_normalized, activation_path, novelty = agent.perceive_and_update_state(cortex_id, current_obs)
                        action, log_prob, _, decision_maker, epsilon, _, action_probs = agent.select_action(actual_action_dim, activation_path)

                        # Update postfix and UI state
                        postfix_data["Epsilon"] = f"{epsilon:.2f}"
                        postfix_data["Decision"] = decision_maker
                        update_ui_state_in_redis('chimera_action_update', {'probabilities': action_probs})

                        await connector.send_action(int(action))
                        msg = await connector.receive_message()

                        if not msg:
                            logger.warning("Did not receive state, might be reconnecting. Ending episode.")
                            break

                        if msg.get("type") == "action.taken":
                            next_obs = np.array(msg.get("observation"))
                            reward = msg.get("reward", 0)
                            done = msg.get("done", False)
                            episode_reward += reward

                            agent.record_experience(h_t, z_t, activation_path, current_obs, action, log_prob, reward, next_obs, done)
                            current_obs = next_obs

                            if agent.steps_done > hyperparams.get('burnin_steps', 1000) and \
                               agent.steps_done % hyperparams.get('policy_train_frequency', 10) == 0 and \
                               len(agent.replay_buffer) > hyperparams.get('batch_size', 16):
                                train_stats = agent.train(cortex_id=cortex_id)
                                if train_stats:
                                    wm_loss = train_stats.get('wm_loss_total')
                                    ac_loss = train_stats.get('ac_loss')
                                    postfix_data["WM Loss"] = f"{wm_loss:.4f}" if isinstance(wm_loss, (int, float)) else "N/A"
                                    postfix_data["AC Loss"] = f"{ac_loss:.4f}" if isinstance(ac_loss, (int, float)) else "N/A"
                                    update_ui_state_in_redis('chimera_training_metrics', train_stats)

                                    updated_graph = agent.get_graph_structure()
                                    postfix_data["Nodes"] = len(updated_graph.get('nodes', []))
                                    postfix_data["Edges"] = len(updated_graph.get('edges', []))
                                    update_ui_state_in_redis('chimera_graph_state', updated_graph)

                        elif msg.get("type") == "game.over":
                            done = True
                        else:
                            logger.warning(f"Unexpected message type received: {msg.get('type')}")

                        pbar.set_postfix(postfix_data)

                    # --- End of Episode ---
                    postfix_data["Reward"] = f"{episode_reward:.2f}"
                    pbar.set_postfix(postfix_data)
                    pbar.update(1)

                    episode_stats = { 'episode': episode, 'reward': episode_reward, 'total_steps': agent.steps_done }
                    update_ui_state_in_redis('chimera_episode_metrics', episode_stats)

                    if episode < episodes:
                        reset_response = await connector.reset_environment()
                        if reset_response:
                            join_response = reset_response
                        else:
                            logger.error("Failed to reset environment. Exiting.")
                            break

        finally:
            logger.info("Closing connection and saving agent state.")
            await connector.close()
            if 'agent' in locals():
                agent.save_state(version_info={"message": f"Colosseum training on {env_id} stopped."})
