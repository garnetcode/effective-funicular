import asyncio
import logging
import yaml
import numpy as np
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
        parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to the main configuration file.")
        parser.add_argument("--env-id", type=str, default="LunarLander-v3", help="The Colosseum environment ID to train in.")
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

    async def _setup(self, options):
        """Sets up the agent, connector, and environment."""
        env_id = options['env_id']
        agent_tag = options['agent_tag'] or f"chimera-agent-{int(time.time())}"
        config_path = options['config']
        port = options['port']

        logger.info(f"Starting Chimera agent '{agent_tag}' for environment: {env_id} on port {port}")

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except FileNotFoundError:
            logger.error(f"Config file not found at: {config_path}")
            return None, None, None, None

        connector = ColosseumConnector(env_id, agent_tag, http_port=port, ws_port=port)
        session_info = await connector.create_session()
        if not session_info:
            logger.error("Could not create session. Exiting.")
            return None, None, None, None

        env_specs = session_info['environment']
        obs_space_info = env_specs['observation_space']
        action_space_info = env_specs['action_space']

        cortex_configs, cortex_id = create_cortex_configs_from_observation_space(
            gym.spaces.Box(
                low=np.array(obs_space_info['low']),
                high=np.array(obs_space_info['high']),
                shape=obs_space_info['shape'],
                dtype=np.dtype(obs_space_info['dtype'])
            )
        )
        actual_action_dim = action_space_info['n']

        agent = ChimeraAgent(
            agent_id=agent_tag,
            embedding_dim=config['agent_config'].get('embedding_dim', 512),
            max_action_dim=actual_action_dim,
            cortex_configs=cortex_configs,
            load_from_storage=not config.get('force_new_agent', False),
            hyperparams=config['agent_config'].get('hyperparams', {}),
            history_config=config.get('agent_history', {})
        )
        agent.set_active_skill(env_id)
        logger.info(f"Initialized Chimera Agent '{agent_tag}' for {env_id}")

        if not await connector.connect_websocket() or not await connector.join_session():
            logger.error("Could not connect to WebSocket or join session. Exiting.")
            await connector.close()
            return None, None, None, None

        return agent, connector, session_info, cortex_id

    async def _run_episode(self, episode_num, agent, connector, initial_obs, cortex_id):
        """Runs a single episode of the training loop."""
        done = False
        episode_reward = 0
        current_obs = initial_obs
        hyperparams = agent.hyperparams
        actual_action_dim = agent.max_action_dim

        step_num = 0
        while not done:
            try:
                step_num += 1
                logger.info(f"[Ep.{episode_num} Step.{step_num}] Perceiving state...")
                # Agent logic
                h_t, z_t, h_normalized, activation_path, novelty = agent.perceive_and_update_state(cortex_id, current_obs)
                logger.info(f"[Ep.{episode_num} Step.{step_num}] Selecting action...")
                action, log_prob, _, _, _, _, action_probs = agent.select_action(actual_action_dim, activation_path)
                update_ui_state_in_redis('chimera_action_update', {'probabilities': action_probs})

                # Network interaction
                logger.info(f"[Ep.{episode_num} Step.{step_num}] Sending action {action}...")
                await connector.send_action(int(action))
                logger.info(f"[Ep.{episode_num} Step.{step_num}] Waiting for message...")
                msg = await connector.receive_message()

                if not msg:
                    logger.warning("Did not receive state from server. Ending episode.")
                    return "CONNECTION_ERROR", episode_reward, agent.steps_done

                if msg.get("type") == "action.taken":
                    next_obs = np.array(msg.get("observation"))
                    reward = msg.get("reward", 0)
                    done = msg.get("done", False)
                    episode_reward += reward

                    agent.record_experience(h_t, z_t, activation_path, current_obs, action, log_prob, reward, next_obs, done)
                    current_obs = next_obs

                    # Potentially blocking training call
                    if agent.steps_done > hyperparams.get('burnin_steps', 1000) and \
                       agent.steps_done % hyperparams.get('policy_train_frequency', 10) == 0 and \
                       len(agent.replay_buffer) > hyperparams.get('batch_size', 16):
                        train_stats = await asyncio.to_thread(agent.train, cortex_id=cortex_id)
                        if train_stats:
                            update_ui_state_in_redis('chimera_training_metrics', train_stats)
                            updated_graph = agent.get_graph_structure()
                            update_ui_state_in_redis('chimera_graph_state', updated_graph)

                elif msg.get("type") == "game.over":
                    done = True
                else:
                    logger.warning(f"Unexpected message type received: {msg.get('type')}")

            except Exception as e:
                logger.error(f"Error during episode {episode_num}: {e}", exc_info=True)
                return "EPISODE_ERROR", episode_reward, agent.steps_done

        return "SUCCESS", episode_reward, agent.steps_done

    async def _run_training_loop(self, agent, connector, session_info, cortex_id, episodes):
        """Runs the main training loop over all episodes."""
        initial_obs = np.array(session_info.get("observation"))
        with tqdm(total=episodes, desc=f"Training on {connector.environment_id}") as pbar:
            for episode in range(1, episodes + 1):
                status, episode_reward, total_steps = await self._run_episode(episode, agent, connector, initial_obs, cortex_id)

                if status != "SUCCESS":
                    logger.error(f"Episode {episode} failed with status: {status}. Aborting training.")
                    break

                pbar.set_postfix({"Last Reward": f"{episode_reward:.2f}", "Total Steps": total_steps})
                pbar.update(1)

                episode_stats = {'episode': episode, 'reward': episode_reward, 'total_steps': total_steps}
                update_ui_state_in_redis('chimera_episode_metrics', episode_stats)

                if episode < episodes:
                    reset_response = await connector.reset_environment()
                    if reset_response:
                        initial_obs = np.array(reset_response.get("observation"))
                    else:
                        logger.error("Failed to reset environment. Exiting training loop.")
                        break

    async def async_handle(self, *args, **options):
        """The main asynchronous logic for training the agent."""
        log_level = logging.INFO if options['verbosity'] > 0 else logging.WARNING
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        agent, connector, session_info, cortex_id = await self._setup(options)

        if not all([agent, connector, session_info, cortex_id]):
            logger.error("Setup failed. Aborting training.")
            return

        try:
            await self._run_training_loop(agent, connector, session_info, cortex_id, options['episodes'])
        finally:
            logger.info("Closing connection and saving agent state.")
            await connector.close()
            agent.save_state(version_info={"message": f"Colosseum training on {connector.environment_id} stopped."})
