import asyncio
import logging
import yaml
import numpy as np
from tqdm import tqdm
import time
import threading

from django.core.management.base import BaseCommand
from api.services.chimera_agent import ChimeraAgent
from colosseum_connector import ColosseumConnector
from api.services.cortex.factory import create_cortex_configs_from_observation_space
from api.services.per_sequence_buffer import PERSequenceBuffer
import gymnasium as gym
import redis
import json

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
    if redis_client is None: return
    try:
        payload = json.dumps(data, cls=NumpyJSONEncoder)
        redis_client.set(key, payload)
    except Exception as e:
        logger.warning(f"Failed to update UI state in Redis for key {key}: {e}")

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class LearnerThread(threading.Thread):
    """A dedicated thread for running the agent's training loop."""
    def __init__(self, agent, cortex_id, stop_event):
        super().__init__()
        self.agent = agent
        self.cortex_id = cortex_id
        self.stop_event = stop_event
        self.daemon = True # Allows main thread to exit even if this thread is still running

    def run(self):
        logger.info("Learner thread started.")
        while not self.stop_event.is_set():
            try:
                hyperparams = self.agent.hyperparams
                if len(self.agent.replay_buffer) > hyperparams.get('burnin_steps', 1000) and \
                   len(self.agent.replay_buffer) > hyperparams.get('batch_size', 16):

                    logger.debug("Learner is training a batch.")
                    # This is a synchronous, blocking call, which is why it's in its own thread.
                    train_stats = self.agent.train(cortex_id=self.cortex_id)
                    if train_stats:
                        update_ui_state_in_redis('chimera_training_metrics', train_stats)
                        updated_graph = self.agent.get_graph_structure()
                        update_ui_state_in_redis('chimera_graph_state', updated_graph)
                else:
                    time.sleep(1) # Avoid busy-waiting
            except Exception as e:
                logger.error(f"Error in learner thread: {e}", exc_info=True)
                time.sleep(5)
        logger.info("Learner thread stopped.")

class Command(BaseCommand):
    help = 'Train a ChimeraAgent using a Colosseum server environment.'

    def add_arguments(self, parser):
        parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to the main configuration file.")
        parser.add_argument("--env-id", type=str, default="LunarLander-v3", help="The Colosseum environment ID to train in.")
        parser.add_argument("--episodes", type=int, default=500, help="Total number of episodes to train for.")
        parser.add_argument("--agent-tag", type=str, default=None, help="A unique tag for the agent.")
        parser.add_argument("--port", type=int, default=8002, help="The port of the Colosseum server.")

    def handle(self, *args, **options):
        try:
            asyncio.run(self.async_handle(*args, **options))
        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in main handler: {e}", exc_info=True)

    async def _setup(self, options):
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
            return None

        hyperparams = config['agent_config'].get('hyperparams', {})
        replay_buffer = PERSequenceBuffer(
            capacity=hyperparams.get('buffer_capacity', 10000),
            sequence_length=hyperparams.get('sequence_length', 50),
            alpha=hyperparams.get('per_alpha', 0.6),
            beta_start=hyperparams.get('per_beta_start', 0.4),
            beta_frames=hyperparams.get('per_beta_frames', 100000)
        )

        connector = ColosseumConnector(env_id, agent_tag, http_port=port, ws_port=port)
        session_info = await connector.create_session()
        if not session_info:
            logger.error("Could not create session.")
            return None

        env_specs = session_info['environment']
        obs_space_info = env_specs['observation_space']
        action_space_info = env_specs['action_space']
        cortex_configs, cortex_id = create_cortex_configs_from_observation_space(
            gym.spaces.Box(
                low=np.array(obs_space_info['low']), high=np.array(obs_space_info['high']),
                shape=obs_space_info['shape'], dtype=np.dtype(obs_space_info['dtype'])
            )
        )
        agent = ChimeraAgent(
            agent_id=agent_tag,
            embedding_dim=config['agent_config'].get('embedding_dim', 512),
            max_action_dim=action_space_info['n'],
            cortex_configs=cortex_configs,
            load_from_storage=not config.get('force_new_agent', False),
            hyperparams=hyperparams,
            history_config=config.get('agent_history', {}),
            replay_buffer=replay_buffer
        )
        agent.set_active_skill(env_id)
        logger.info(f"Initialized Chimera Agent '{agent_tag}' for {env_id}")

        if not await connector.connect_websocket() or not await connector.join_session():
            logger.error("Could not connect to WebSocket or join session.")
            await connector.close()
            return None

        return agent, connector, session_info, cortex_id

    async def _actor_task(self, agent, connector, session_info, cortex_id, episodes):
        logger.info("Actor task started.")
        initial_obs = np.array(session_info.get("observation"))
        with tqdm(total=episodes, desc=f"Acting in {connector.environment_id}") as pbar:
            for episode in range(1, episodes + 1):
                done = False
                episode_reward = 0
                current_obs = initial_obs

                while not done:
                    h_t, z_t, h_normalized, activation_path, novelty = agent.perceive_and_update_state(cortex_id, current_obs)
                    action, log_prob, _, _, _, _, action_probs = agent.select_action(agent.max_action_dim, activation_path)
                    update_ui_state_in_redis('chimera_action_update', {'probabilities': action_probs})

                    await connector.send_action(int(action))
                    msg = await connector.receive_message()

                    if not msg:
                        logger.warning("Did not receive state from server. Ending episode.")
                        break

                    if msg.get("type") == "action.taken":
                        next_obs = np.array(msg.get("observation"))
                        reward = msg.get("reward", 0)
                        done = msg.get("done", False)
                        episode_reward += reward
                        agent.record_experience(h_t, z_t, activation_path, current_obs, action, log_prob, reward, next_obs, done)
                        current_obs = next_obs
                    elif msg.get("type") == "game.over":
                        done = True
                    else:
                        logger.warning(f"Unexpected message type received: {msg.get('type')}")

                pbar.set_postfix({"Last Reward": f"{episode_reward:.2f}", "Total Steps": agent.steps_done})
                pbar.update(1)
                update_ui_state_in_redis('chimera_episode_metrics', {'episode': episode, 'reward': episode_reward, 'total_steps': agent.steps_done})

                if episode < episodes:
                    reset_response = await connector.reset_environment()
                    if reset_response:
                        initial_obs = np.array(reset_response.get("observation"))
                    else:
                        logger.error("Failed to reset environment. Stopping actor task.")
                        break
        logger.info("Actor task finished.")

    async def async_handle(self, *args, **options):
        log_level = logging.INFO if options.get('verbosity', 0) > 0 else logging.WARNING
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        setup_result = await self._setup(options)
        if not setup_result:
            logger.error("Setup failed. Aborting training.")
            return

        agent, connector, session_info, cortex_id = setup_result
        stop_event = threading.Event()

        learner_thread = LearnerThread(agent=agent, cortex_id=cortex_id, stop_event=stop_event)
        learner_thread.start()

        try:
            await self._actor_task(agent, connector, session_info, cortex_id, options['episodes'])
        finally:
            logger.info("Actor finished. Stopping learner and cleaning up.")
            stop_event.set()
            learner_thread.join() # Wait for the learner thread to finish
            await connector.close()
            agent.save_state(version_info={"message": f"Colosseum training on {connector.environment_id} stopped."})
