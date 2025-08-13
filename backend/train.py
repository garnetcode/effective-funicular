import numpy as np
import os
import argparse
from api.services.chimera_agent import ChimeraAgent

# NOTE: This script requires the `gymnasium` library.
# It was written assuming the library is installed, but the environment
# prevented the installation. To run this, please ensure you have run:
# pip install gymnasium[all]

try:
    import gymnasium as gym
except ImportError:
    print("FATAL: gymnasium library not found. Please install it with `pip install gymnasium[all]`")
    exit(1)


def get_env_config(env):
    """Inspects a gymnasium environment to determine agent configuration."""

    # Observation space -> Cortex config
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box):
        if len(obs_space.shape) != 1:
            raise NotImplementedError("Only 1D Box observation spaces are supported for now (e.g., classic control).")
        input_dim = obs_space.shape[0]
        cortex_configs = {
            "vector_input": {
                "type": "DenseCortex",
                "params": {"input_dim": input_dim}
            }
        }
        cortex_id = "vector_input"
    else:
        raise NotImplementedError(f"Observation space type {type(obs_space)} not supported yet.")

    # Action space -> Action head config
    act_space = env.action_space
    if isinstance(act_space, gym.spaces.Discrete):
        n_actions = act_space.n
    else:
        raise NotImplementedError(f"Action space type {type(act_space)} not supported yet.")

    return cortex_configs, cortex_id, n_actions


def main(args):
    """
    Main function to run the RL training loop with a gymnasium environment.
    """
    print(f"Initializing environment: {args.env}")
    env = gym.make(args.env)

    # --- Dynamic Configuration ---
    cortex_configs, cortex_id, n_actions = get_env_config(env)

    # --- Agent Initialization ---
    agent_id = args.agent_id or f"agent-{args.env}"

    agent = ChimeraAgent(
        agent_id=agent_id,
        dimensions=args.dims,
        n_actions=n_actions,
        cortex_configs=cortex_configs,
        load_from_storage=not args.force_new,
        hyperparams={'learning_rate': args.lr, 'gamma': args.gamma}
    )

    print(f"Starting training for agent '{agent_id}' in '{args.env}'...")
    print(f"Agent brain dimensions: {args.dims}, Actions: {n_actions}")

    # --- Training Loop ---
    total_rewards = []

    for episode in range(args.episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        while not (terminated or truncated):
            # 1. Perceive the state
            state_embedding = agent.perceive(cortex_id, state)

            # 2. Select an action
            action, log_prob, internal_state = agent.select_action(state_embedding)

            # 3. Take action in the environment
            next_state, reward, terminated, truncated, info = env.step(action)

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
            print(f"Episode {episode}/{args.episodes} | Total Reward: {episode_reward:.2f} | Avg Reward (last 100): {avg_reward:.2f}")

    env.close()
    print("Training finished.")
    print(f"Final agent state saved to {agent.storage_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ChimeraAgent in a gymnasium environment.")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Name of the gymnasium environment.")
    parser.add_argument("--agent_id", type=str, default=None, help="A unique ID for the agent. Defaults to 'agent-<env_name>'.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train for.")
    parser.add_argument("--dims", type=int, default=64, help="Dimensionality of the agent's internal brain space.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for the agent.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards.")
    parser.add_argument("--force_new", action="store_true", help="Force creation of a new agent, ignoring saved state.")

    args = parser.parse_args()
    main(args)
