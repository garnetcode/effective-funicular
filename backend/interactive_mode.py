import yaml
import logging
import numpy as np
import torch
from api.services.chimera_agent import ChimeraAgent

# --- Set up logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

    # --- Check if language model is enabled ---
    if not config.get('language_model', {}).get('enabled', False):
        logger.error("Language model is not enabled in the configuration file. Exiting.")
        return

    # --- Agent Initialization ---
    # For interactive mode, we need a dummy observation and action dimension,
    # as the agent will only be processing text from the language cortex.
    obs_dim = config['agent_config'].get('latent_dim', 64) # The language embedding projects to this size
    action_dim = 1 # Dummy action space

    agent_id = config.get('default_agent_id_prefix', 'agent-') + "interactive"
    agent_config = config.get('agent_config', {})
    agent_config['hyperparams'] = {**agent_config.get('hyperparams', {}), **config.get('language_model', {})}

    agent = ChimeraAgent(
        agent_id=agent_id,
        obs_dim=obs_dim,
        action_dim=action_dim,
        cortex_configs={}, # Cortexes will be configured by the agent based on hyperparams
        load_from_storage=True, # Try to load a pre-trained agent
        **agent_config
    )

    logger.info(f"Agent '{agent_id}' loaded for interactive mode.")
    logger.info("You can now chat with the agent. Type 'quit' to exit.")

    # --- Interactive Loop ---
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # 1. Agent perceives the text input
        agent.perceive_and_update_state('language_cortex', user_input)

        # 2. Agent generates a response based on its new internal state
        response = agent.generate_response()

        print(f"Agent: {response}")

if __name__ == "__main__":
    main()
