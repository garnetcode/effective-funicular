import asyncio
import logging
import yaml
import numpy as np
from tqdm import tqdm
import time
import threading
import copy

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

class SharedModelWeights:
    """A thread-safe class to hold the model weights for the actor to pull."""
    def __init__(self):
        self.state_dict = None
        self.lock = threading.Lock()

    def set_weights(self, state_dict):
        with self.lock:
            self.state_dict = copy.deepcopy(state_dict)

    def get_weights(self):
        with self.lock:
            return copy.deepcopy(self.state_dict) if self.state_dict else None

class LearnerThread(threading.Thread):
    """A dedicated thread for running the agent's training loop."""
    def __init__(self, agent, cortex_id, env_id, stop_event, shared_weights):
        super().__init__()
        self.agent = agent
        self.cortex_id = cortex_id
        self.env_id = env_id
        self.stop_event = stop_event
        self.shared_weights = shared_weights
        self.daemon = True

    def run(self):
        logger.info("Learner thread started.")
        train_steps = 0
        while not self.stop_event.is_set():
            try:
                hyperparams = self.agent.hyperparams
                if len(self.agent.replay_buffer) > hyperparams.get('burnin_steps', 1000):
                    train_stats = self.agent.train(cortex_id=self.cortex_id)
                    train_steps += 1

                    if train_stats:
                        update_ui_state_in_redis('chimera_training_metrics', train_stats)

                        # Log training stats periodically to the console
                        if train_steps % 20 == 0: # Log every 20 training steps
                            stats_str = ", ".join([f"{key}={value:.4f}" for key, value in train_stats.items() if isinstance(value, (int, float))])
                            logger.info(f"Learner (train_step {train_steps}): {stats_str}")

                    if train_steps % hyperparams.get('actor_update_frequency', 100) == 0:
                        logger.info(f"Learner publishing new weights for actor at step {train_steps}.")
                        self.shared_weights.set_weights(self.agent.get_actor_state_dict())

                    # Periodically update the STAG graph in the UI
                    if train_steps % hyperparams.get('stag_ui_update_frequency', 50) == 0:
                        stag_graph = self.agent.get_graph_structure(self.env_id)
                        if stag_graph:
                            update_ui_state_in_redis('chimera_stag_graph', stag_graph)
                else:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in learner thread: {e}", exc_info=True)
                time.sleep(5)
        logger.info("Learner thread stopped.")

class Command(BaseCommand):
    help = 'Train a ChimeraAgent using a Colosseum server environment.'

    def add_arguments(self, parser):
        parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to the main configuration file.")
        parser.add_argument("--env-id", type=str, default="LunarLander-v3", help="The Colosseum environment ID to train in.")
        parser.add_argument("--agent-tag", type=str, default=None, help="A unique tag for the agent.")
        parser.add_argument("--port", type=int, default=8002, help="The port of the Colosseum server.")

    def handle(self, *args, **options):
        try:
            asyncio.run(self.async_handle(*args, **options))
        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in main handler: {e}", exc_info=True)

    async def _setup_components(self, options):
        env_id = options['env_id']
        # AGENT_FIX: Use the env_id to create a persistent agent name, allowing history to be loaded.
        agent_tag = options['agent_tag'] or f"chimera-agent-{env_id}"
        config_path = options['config']
        port = options['port']

        logger.info(f"Setting up components for agent '{agent_tag}' in '{env_id}'...")

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at: {config_path}")
            return None

        hyperparams = config.get('agent_config', {}).get('hyperparams', {})
        replay_buffer = PERSequenceBuffer(
            capacity=hyperparams.get('buffer_capacity', 10000),
            sequence_length=hyperparams.get('sequence_length', 50),
            alpha=hyperparams.get('per_alpha', 0.6),
            beta_start=hyperparams.get('per_beta_start', 0.4),
            beta_frames=hyperparams.get('per_beta_frames', 100000)
        )

        connector = ColosseumConnector(env_id, agent_tag, http_port=port, ws_port=port)
        session_info = await connector.create_session()
        if not session_info: return None

        env_specs = session_info['environment']
        obs_space_info = env_specs['observation_space']
        action_space_info = env_specs['action_space']

        cortex_configs, cortex_id = create_cortex_configs_from_observation_space(
            gym.spaces.Box(
                low=np.array(obs_space_info['low']), high=np.array(obs_space_info['high']),
                shape=obs_space_info['shape'], dtype=np.dtype(obs_space_info['dtype'])
            )
        )

        common_params = {
            "embedding_dim": config.get('agent_config', {}).get('embedding_dim', 512),
            "max_action_dim": action_space_info['n'],
            "cortex_configs": cortex_configs,
            "hyperparams": hyperparams,
            "replay_buffer": replay_buffer
        }

        learner_agent = ChimeraAgent(
            agent_id=agent_tag,
            load_from_storage=not config.get('force_new_agent', False),
            history_config=config.get('agent_history', {}),
            **common_params
        )

        actor_agent = ChimeraAgent(
            agent_id=f"{agent_tag}-actor",
            load_from_storage=False,
            history_config={},
            **common_params
        )
        actor_agent.load_state_dict(learner_agent.state_dict())

        learner_agent.set_active_skill(env_id)
        actor_agent.set_active_skill(env_id)

        shared_weights = SharedModelWeights()
        shared_weights.set_weights(learner_agent.get_actor_state_dict())

        if not await connector.connect_websocket() or not await connector.join_session():
            await connector.close()
            return None

        logger.info("Setup complete. Actor and Learner are ready.")
        return learner_agent, actor_agent, connector, session_info, cortex_id, shared_weights

    async def _actor_task(self, actor_agent, shared_weights, connector, session_info, cortex_id):
        logger.info("Actor task started.")
        current_obs = np.array(session_info.get("observation"))
        best_episode_reward = -float('inf')
        episode = 0

        while True: # Run indefinitely until interrupted by KeyboardInterrupt
            episode += 1
            logger.info(f"--- Starting Episode {episode} ---")
            latest_weights = shared_weights.get_weights()
            if latest_weights:
                actor_agent.load_actor_state_dict(latest_weights)

            done = False
            episode_reward = 0

            while not done:
                h_t, z_t, h_normalized, activation_path, novelty = actor_agent.perceive_and_update_state(cortex_id, current_obs)
                action, log_prob, _, _, _, _, action_probs = actor_agent.select_action(actor_agent.max_action_dim, activation_path)

                update_ui_state_in_redis('chimera_action_update', {'probabilities': action_probs})
                await connector.send_action(int(action))
                msg = await connector.receive_message()

                if not msg:
                    logger.warning("Actor did not receive state from server. Ending episode.")
                    break

                if msg.get("type") == "action.taken":
                    next_obs = np.array(msg.get("observation"))
                    reward = msg.get("reward", 0)
                    done = msg.get("done", False)
                    episode_reward += reward
                    actor_agent.record_experience(h_t, z_t, activation_path, current_obs, action, log_prob, reward, next_obs, done)
                    current_obs = next_obs
                elif msg.get("type") == "game.over":
                    logger.info("Game over message received. Ending episode.")
                    reward = msg.get("reward", 0)
                    episode_reward += reward
                    done = True
                    actor_agent.record_experience(h_t, z_t, activation_path, current_obs, action, log_prob, reward, current_obs, done)
                else:
                    logger.warning(f"Actor received unexpected message type: {msg.get('type')}")

            logger.info(f"Episode {episode} finished. Reward: {episode_reward:.2f}, Total Steps: {actor_agent.steps_done}")
            update_ui_state_in_redis('chimera_episode_metrics', {'episode': episode, 'reward': episode_reward, 'total_steps': actor_agent.steps_done})

            # Save the model if it's the best one seen so far.
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                logger.info(f"New best reward: {best_episode_reward:.2f}. Saving model...")
                actor_agent.save_state(version_info={"message": f"New best model with reward {best_episode_reward:.2f} at episode {episode}."})

            reset_response = await connector.reset_environment()
            if reset_response:
                current_obs = np.array(reset_response.get("observation"))
            else:
                logger.error("Failed to reset environment. Stopping actor task.")
                break
        logger.info("Actor task finished.")

    async def async_handle(self, *args, **options):
        log_level = logging.INFO if options.get('verbosity', 0) > 0 else logging.WARNING
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        setup_result = await self._setup_components(options)
        if not setup_result:
            logger.error("Setup failed. Aborting training.")
            return

        learner_agent, actor_agent, connector, session_info, cortex_id, shared_weights = setup_result
        stop_event = threading.Event()

        learner_thread = LearnerThread(
            agent=learner_agent,
            cortex_id=cortex_id,
            env_id=connector.environment_id,
            stop_event=stop_event,
            shared_weights=shared_weights
        )

        try:
            learner_thread.start()
            await self._actor_task(actor_agent, shared_weights, connector, session_info, cortex_id)
        finally:
            logger.info("Main task finished. Stopping learner and cleaning up.")
            stop_event.set()
            learner_thread.join()
            await connector.close()
            logger.info("Training complete. Best model was saved during the run based on episode performance.")
