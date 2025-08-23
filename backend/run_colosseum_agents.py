import asyncio
import logging
import yaml
from multi_session_manager import MultiSessionManager

# --- Set up logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("colosseum_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # --- Load Configuration ---
    config_path = "backend/config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at: {config_path}")
        return

    logger.info(f"Loaded configuration from {config_path}")

    # Environments to run in parallel
    # This could also be moved into the config file
    env_list = [
        "CartPole-v1",
        "LunarLander-v2" # Note: The spec mentioned v3, but v2 is also common
    ]

    # Agent configuration (hyperparameters, etc.)
    agent_config = config.get('agent_config', {})
    agent_config['load_from_storage'] = not config.get('force_new_agent', False)
    history_config = config.get('agent_history', {})
    num_episodes = config.get('episodes_per_env', 10)

    # --- Start the Manager ---
    manager = MultiSessionManager(
        agent_config=agent_config,
        history_config=history_config,
        env_list=env_list,
        num_episodes=num_episodes
    )

    logger.info(f"Starting Multi-Session Manager for {num_episodes} episodes in environments: {env_list}")
    try:
        asyncio.run(manager.start())
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        logger.info("Training finished.")

if __name__ == "__main__":
    main()
