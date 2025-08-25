import asyncio
import logging
import yaml
import numpy as np
import torch

from api.services.chimera_agent import ChimeraAgent
from colosseum_connector import ColosseumConnector
from api.services.cortex.factory import create_cortex_configs_from_observation_space
import gymnasium as gym

# --- Set up logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("colosseum_curriculum.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def get_env_specs(env_id):
    """Connects to the server to get environment specs and then disconnects."""
    logger.info(f"Inspecting environment: {env_id}")
    connector = ColosseumConnector(env_id, "spec-inspector")
    session_data = await connector.create_session()
    if not session_data:
        logger.error(f"Could not get specs for {env_id}")
        return None
    # We don't need to maintain the connection, just get the specs
    return session_data['environment']

async def run_training_curriculum():
    # --- Load Configuration ---
    config_path = "config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at: {config_path}")
        return

    logger.info(f"Loaded configuration from {config_path}")

    # --- Curriculum and Agent Setup ---
    # Environments to run sequentially as a curriculum
    env_list = [
        "CartPole-v1",
        "LunarLander-v2"
    ]
    episodes_per_env = config.get('episodes_per_env', 10)
    agent_config = config.get('agent_config', {})
    history_config = config.get('agent_history', {})

    # --- Pre-inspect all environments by connecting to the server ---
    master_cortex_configs = {}
    max_action_dim = 0
    valid_env_list = []
    for name in env_list:
        specs = await get_env_specs(name)
        if specs:
            try:
                # Reconstruct the observation space from the server's response
                obs_space_info = specs['observation_space']
                observation_space = gym.spaces.Box(
                    low=np.array(obs_space_info['low']),
                    high=np.array(obs_space_info['high']),
                    shape=obs_space_info['shape'],
                    dtype=np.dtype(obs_space_info['dtype'])
                )

                env_cortex_configs, _ = create_cortex_configs_from_observation_space(observation_space)
                master_cortex_configs.update(env_cortex_configs)

                # Reconstruct the action space
                action_space_info = specs['action_space']
                if action_space_info.get('type', '').lower() == 'discrete':
                    max_action_dim = max(max_action_dim, action_space_info.get('n', 0))

                valid_env_list.append(name)
            except (KeyError, TypeError) as e:
                logger.error(f"Could not parse specs for environment {name}: {e}. Skipping.")

    env_list = valid_env_list
    logger.info(f"Master cortex configuration will include: {list(master_cortex_configs.keys())}")
    logger.info(f"Max action dimension across all environments: {max_action_dim}")

    # --- Initialize a single, generalist agent ---
    agent_id = agent_config.get('default_agent_id_prefix', 'Kymera-Generalist') + "v1"
    embedding_dim = agent_config.get('embedding_dim', 512)

    agent = ChimeraAgent(
        agent_id=agent_id,
        embedding_dim=embedding_dim,
        max_action_dim=max_action_dim,
        cortex_configs=master_cortex_configs,
        load_from_storage=not config.get('force_new_agent', False),
        hyperparams=agent_config.get('hyperparams', {}),
        history_config=history_config
    )
    logger.info(f"Initialized Generalist Agent '{agent_id}'")

    # --- Run the Curriculum ---
    for env_id in env_list:
        logger.info(f"--- Starting Curriculum Stage: {env_id} for {episodes_per_env} episodes ---")
        agent.set_active_skill(env_id)

        connector = ColosseumConnector(env_id, agent.agent_id)

        try:
            # Create session and connect
            session_data = await connector.create_session()
            if not session_data: continue

            if not await connector.connect_websocket(): continue

            join_response = await connector.join_session()
            if not join_response: continue

            # Get env-specific details from the session data
            current_obs = np.array(session_data.get("observation"))
            env_specs = session_data['environment']
            actual_action_dim = env_specs['action_space']['n']

            # Reconstruct observation space to determine the correct cortex_id
            obs_space_info = env_specs['observation_space']
            observation_space = gym.spaces.Box(
                low=np.array(obs_space_info['low']),
                high=np.array(obs_space_info['high']),
                shape=obs_space_info['shape'],
                dtype=np.dtype(obs_space_info['dtype'])
            )
            _, cortex_id = create_cortex_configs_from_observation_space(observation_space)

            # Run episodes for this environment
            for episode in range(episodes_per_env):
                done = False
                episode_reward = 0

                while not done:
                    # Perceive, act, and learn
                    h_t, z_t, h_normalized, activation_path, novelty = agent.perceive_and_update_state(cortex_id, current_obs)
                    action, log_prob, _, _, _ = agent.select_action(actual_action_dim, activation_path)

                    await connector.send_action(action)
                    msg = await connector.receive_message()

                    if not msg:
                        logger.warning(f"[{agent.agent_id}] Disconnection detected. Ending episode.")
                        done = True
                        continue

                    if msg.get("type") == "action.taken":
                        next_obs = np.array(msg.get("observation"))
                        reward = msg.get("reward")
                        done = msg.get("done")

                        agent.record_experience(h_t, z_t, activation_path, current_obs, action, log_prob, reward, next_obs, done)
                        current_obs = next_obs
                        episode_reward += reward

                    elif msg.get("type") == "game.over":
                        done = True
                    else:
                        logger.warning(f"Unexpected message type: {msg.get('type')}")

                # Post-episode training
                train_stats = agent.train(cortex_id=cortex_id)
                logger.info(f"[{agent.agent_id}][{env_id}] Ep {episode+1}/{episodes_per_env} | Reward: {episode_reward:.2f} | Policy Loss: {train_stats.get('policy_loss', 'N/A'):.4f}")

                # Reset for next episode
                if episode < episodes_per_env - 1:
                    reset_response = await connector.reset_environment()
                    if reset_response:
                        current_obs = np.array(reset_response.get("observation"))
                    else:
                        logger.error(f"[{agent.agent_id}] Failed to reset env. Moving to next curriculum stage.")
                        break

        except Exception as e:
            logger.error(f"An error occurred during training on {env_id}: {e}", exc_info=True)
        finally:
            await connector.close()
            logger.info(f"--- Finished Curriculum Stage: {env_id} ---")

    agent.save_state(version_info={"message": "Curriculum training complete."})
    logger.info("Full curriculum training finished.")

if __name__ == "__main__":
    try:
        asyncio.run(run_training_curriculum())
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        logger.info("Script finished.")
