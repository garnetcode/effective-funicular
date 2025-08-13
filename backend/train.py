import numpy as np
import os
from api.services.chimera_agent import ChimeraAgent
from api.environments.grid_world import GridWorld

def main():
    """
    Main function to run the RL training loop.
    """
    # --- Configuration ---
    N_EPISODES = 500
    AGENT_ID = "gridworld-agent-01"
    ENV_SIZE = 5
    BRAIN_DIMENSIONS = 32
    N_ACTIONS = 4 # GridWorld has 4 actions; agent will ignore the rest

    # --- Initialization ---

    # 1. Initialize the Environment
    env = GridWorld(size=ENV_SIZE)

    # 2. Define the Agent's Cortex Configuration
    # We will flatten the grid state and use a DenseCortex.
    cortex_configs = {
        "grid_input": {
            "type": "DenseCortex",
            "params": {
                "input_dim": ENV_SIZE * ENV_SIZE
            }
        }
    }

    # 3. Initialize the Agent
    # If a saved state exists for this agent, it will be loaded.
    # Otherwise, a new agent will be created and saved.
    agent = ChimeraAgent(
        agent_id=AGENT_ID,
        dimensions=BRAIN_DIMENSIONS,
        n_actions=N_ACTIONS,
        cortex_configs=cortex_configs,
        load_from_storage=True, # Set to False to force a new agent
        hyperparams={'learning_rate': 0.005, 'gamma': 0.99}
    )

    print(f"Starting training for agent '{AGENT_ID}'...")

    # --- Training Loop ---
    total_rewards = []

    for episode in range(N_EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # 1. Perceive the state
            state_flat = state.flatten()
            state_embedding = agent.perceive("grid_input", state_flat)

            # 2. Select an action
            action, log_prob, internal_state = agent.select_action(state_embedding)

            # 3. Take action in the environment
            next_state, reward, done, _ = env.step(action)

            # 4. Record the experience
            agent.record_experience(internal_state, action, reward)

            # 5. Update state and total reward
            state = next_state
            episode_reward += reward

        # End of episode: Train the agent
        agent.train()

        total_rewards.append(episode_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(f"Episode {episode}/{N_EPISODES} | Total Reward: {episode_reward:.2f} | Avg Reward (last 100): {avg_reward:.2f}")

    print("Training finished.")
    # The final trained agent state is automatically saved after the last train() call.
    print(f"Final agent state saved to {agent.storage_path}")


if __name__ == "__main__":
    # This script needs to be run from the root directory of the project
    # so that the paths to services and storage work correctly.
    # e.g., `python backend/train.py`
    main()
