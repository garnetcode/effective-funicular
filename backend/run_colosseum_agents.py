import logging
import yaml
import numpy as np
import torch
import gymnasium as gym

from api.services.chimera_agent import ChimeraAgent
from api.services.cortex.factory import create_cortex_configs_from_observation_space

# --- Set up logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("local_curriculum.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_training_curriculum():
    # --- Load Configuration ---
    config_path = "backend/configs/base.yaml" # Correct path to config
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
    hyperparams = agent_config.get('hyperparams', {})

    # --- Pre-inspect all environments locally ---
    master_cortex_configs = {}
    max_action_dim = 0
    for name in env_list:
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
            env_list.remove(name)

    logger.info(f"Master cortex configuration will include: {list(master_cortex_configs.keys())}")
    logger.info(f"Max action dimension across all environments: {max_action_dim}")

    # --- Initialize a single, generalist agent ---
    agent_id = agent_config.get('default_agent_id_prefix', 'Kymera-Generalist') + "v1-local"
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

    # --- Run the Curriculum ---
    total_steps = 0
    for env_id in env_list:
        logger.info(f"--- Starting Curriculum Stage: {env_id} for {episodes_per_env} episodes ---")
        agent.set_active_skill(env_id)

        try:
            env = gym.make(env_id)
            _, cortex_id = create_cortex_configs_from_observation_space(env.observation_space)
            actual_action_dim = env.action_space.n

            # Run episodes for this environment
            for episode in range(episodes_per_env):
                state, _ = env.reset()
                done = False
                truncated = False
                episode_reward = 0

                while not (done or truncated):
                    total_steps += 1
                    # Perceive, act, and learn
                    h_t, z_t, h_normalized, activation_path, novelty = agent.perceive_and_update_state(cortex_id, state)
                    action, log_prob, _, _, _, _ = agent.select_action(actual_action_dim, activation_path)

                    next_state, reward, done, truncated, info = env.step(action)

                    agent.update_stag(h_normalized, reward)
                    agent.record_experience(h_t, z_t, activation_path, state, action, log_prob, reward, next_state, done or truncated)

                    state = next_state
                    episode_reward += reward

                    # Online Training
                    policy_train_frequency = hyperparams.get('policy_train_frequency', 10)
                    if total_steps > hyperparams.get('burnin_steps', 1000) and \
                       total_steps % policy_train_frequency == 0 and \
                       len(agent.replay_buffer) > hyperparams.get('batch_size', 16):
                        train_stats = agent.train(cortex_id)
                        if train_stats:
                            logger.info(f"  Train Step {agent.train_steps} | AC Loss: {train_stats.get('ac_loss', 'N/A'):.4f}")

                logger.info(f"[{agent.agent_id}][{env_id}] Ep {episode+1}/{episodes_per_env} | Reward: {episode_reward:.2f}")

        except Exception as e:
            logger.error(f"An error occurred during training on {env_id}: {e}", exc_info=True)
        finally:
            if 'env' in locals():
                env.close()
            logger.info(f"--- Finished Curriculum Stage: {env_id} ---")

    agent.save_state(version_info={"message": "Local curriculum training complete."})
    logger.info("Full local curriculum training finished.")

if __name__ == "__main__":
    try:
        run_training_curriculum()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        logger.info("Script finished.")
