import asyncio
import logging
import yaml
import numpy as np
import torch
import websockets

from api.services.chimera_agent import ChimeraAgent
from colosseum_connector import ColosseumConnector
from api.services.cortex.factory import create_cortex_configs_from_observation_space
import gymnasium as gym

# --- Set up logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("colosseum_chimera.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def run_chimera_agent(env_id, agent_tag, config_path, episodes):
    """
    Runs the ChimeraAgent in a specified Colosseum environment for a given number of episodes.
    """
    logger.info(f"Starting Chimera agent '{agent_tag}' for environment: {env_id}")

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
    # Use port 8000 to match the default Colosseum server port
    connector = ColosseumConnector(env_id, agent_tag, port=8000)

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
    # Now that we are connected, we have the definitive specs for the environment
    env_specs = join_response['environment']
    obs_space_info = env_specs['observation_space']
    action_space_info = env_specs['action_space']

    # Reconstruct observation space to create cortex configs
    observation_space = gym.spaces.Box(
        low=np.array(obs_space_info['low']),
        high=np.array(obs_space_info['high']),
        shape=obs_space_info['shape'],
        dtype=np.dtype(obs_space_info['dtype'])
    )
    cortex_configs, cortex_id = create_cortex_configs_from_observation_space(observation_space)

    # Get the precise action dimension for this environment
    actual_action_dim = action_space_info['n']

    # Initialize the agent with the environment-specific details
    agent = ChimeraAgent(
        agent_id=agent_tag,
        embedding_dim=agent_config.get('embedding_dim', 512),
        max_action_dim=actual_action_dim, # Use the actual dim for the planner
        cortex_configs=cortex_configs,
        load_from_storage=not config.get('force_new_agent', False),
        hyperparams=hyperparams,
        history_config=history_config
    )
    agent.set_active_skill(env_id, actual_action_dim)
    logger.info(f"Initialized Chimera Agent '{agent_tag}' for {env_id}")

    # --- Main Training Loop ---
    try:
        for episode in range(1, episodes + 1):
            logger.info(f"--- Starting Episode {episode}/{episodes} ---")

            # The initial observation is in the join_response
            current_obs = np.array(join_response.get("observation"))
            done = False
            total_reward = 0

            while not done:
                # 1. Perceive and Select Action
                h_t, z_t, h_normalized, activation_path, novelty = agent.perceive_and_update_state(cortex_id, current_obs)
                action, log_prob, _, _, _, _, _ = agent.select_action(actual_action_dim, activation_path)

                # 2. Step the environment
                await connector.send_action(int(action))
                msg = await connector.receive_message()

                if not msg:
                    logger.warning("Did not receive state, might be reconnecting. Ending episode.")
                    break

                if msg.get("type") == "action.taken":
                    next_obs = np.array(msg.get("observation"))
                    reward = msg.get("reward", 0)
                    done = msg.get("done", False)
                    total_reward += reward

                    # 3. Learn from experience
                    agent.record_experience(h_t, z_t, activation_path, current_obs, action, log_prob, reward, next_obs, done)
                    current_obs = next_obs

                    # 4. Online Training
                    if agent.steps_done > hyperparams.get('burnin_steps', 1000) and \
                       agent.steps_done % hyperparams.get('policy_train_frequency', 10) == 0 and \
                       len(agent.replay_buffer) > hyperparams.get('batch_size', 16):
                        agent.train(cortex_id=cortex_id)

                elif msg.get("type") == "game.over":
                    logger.info(f"Game over message received! Final Score: {msg.get('final_reward')}")
                    done = True
                else:
                    logger.warning(f"Unexpected message type received: {msg.get('type')}")

            logger.info(f"--- Episode {episode} Finished | Total Reward: {total_reward:.2f} ---")

            # 5. Reset for next episode
            if episode < episodes:
                logger.info("Resetting environment for next episode...")
                reset_response = await connector.reset_environment()
                if reset_response:
                    join_response = reset_response # The reset response contains the new initial state
                else:
                    logger.error("Failed to reset environment. Exiting.")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info("Closing connection and saving agent state.")
        await connector.close()
        agent.save_state(version_info={"message": f"Colosseum training on {env_id} stopped."})

if __name__ == "__main__":
    # --- Configuration ---
    ENV_ID = "LunarLander-v2"
    AGENT_TAG = f"chimera-agent-{random.randint(1000, 9999)}"
    CONFIG_PATH = "backend/configs/base.yaml"
    EPISODES_TO_RUN = 500

    try:
        asyncio.run(run_chimera_agent(ENV_ID, AGENT_TAG, CONFIG_PATH, EPISODES_TO_RUN))
    except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError):
        logger.error("\nConnection to the server failed. Please ensure the Colosseum backend is running.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
