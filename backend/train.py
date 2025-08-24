import numpy as np
import os
import argparse
import asyncio
import uuid
import torch
from api.services.chimera_agent import ChimeraAgent
from colosseum_connector import ColosseumConnector

# NOTE: This script requires the `gymnasium` and `websockets` libraries.
# It was written assuming the libraries are installed. To run this, please ensure you have run:
# pip install -r requirements.txt

try:
    import gymnasium as gym
except ImportError:
    print("FATAL: gymnasium library not found. Please install it with `pip install gymnasium[all]`")
    exit(1)


def get_env_config(env):
    """Inspects a gymnasium environment to determine agent configuration."""
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 1:
        input_dim = obs_space.shape[0]
        cortex_configs = {
            "vector_input": {
                "type": "DenseCortex",
                "params": {"input_dim": input_dim}
            }
        }
        cortex_id = "vector_input"
    else:
        raise NotImplementedError(f"Observation space type {type(obs_space)} not supported yet for auto-config.")
    return cortex_configs, cortex_id


def main(args):
    """
    Main function to run the RL training loop with a local environment,
    implementing the Wake-Sleep cycle.
    """
    # --- Environment and Agent Setup ---
    print(f"Creating local environment: {args.env}")
    env = gym.make(args.env)
    cortex_configs, cortex_id = get_env_config(env)
    obs_space_shape = env.observation_space.shape[0]
    actual_action_dim = env.action_space.n

    agent_id = args.agent_id or f"agent-{args.env}"
    # Hyperparameters for the agent and training loop
    hyperparams = {
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'batch_size': args.batch_size,
        'imagine_horizon': 15, # Dreamer paper default
        'collect_interval': 100, # Dreamer paper default
        'train_steps': 10, # Simplified for this example
        'world_model_lr': 1e-3,
        'actor_critic_lr': 3e-4,
        'w_recon': 1.0, 'w_reward': 1.0, 'w_kl': 1.0,
        'w_policy': 1.0, 'w_critic': 0.5, 'w_entropy': 1e-4,
        'lambda': 0.95,
        'use_stag_in_ac_loss': not args.no_stag
    }
    agent = ChimeraAgent(
        agent_id=agent_id,
        max_obs_dim=obs_space_shape, # Set to actual obs dim
        max_action_dim=256, # Keep a max for the model architecture
        cortex_configs=cortex_configs,
        load_from_storage=not args.force_new,
        hyperparams=hyperparams
    )
    print(f"Agent '{agent_id}' configured for '{args.env}'.")

    # --- Main Training Loop (Wake-Sleep Cycle) ---
    total_steps = 0
    episode_num = 0
    try:
        while total_steps < args.total_steps:
            # --- Wake Phase: Collect Experience ---
            print(f"\n--- Wake Phase: Collecting experience ---")
            steps_collected = 0
            while steps_collected < 100: # Collect 100 steps per wake phase
                episode_num += 1
                print(f"--- Episode {episode_num} ---")
                episode_reward = 0

                # Reset environment and agent state
                state, _ = env.reset()
                agent.hidden_state, agent.latent_state = agent.world_model.get_initial_state()
                agent.last_action = torch.tensor([0]) # Reset last action

                terminated = False
                truncated = False

                while not (terminated or truncated):
                    # 1. Perceive, update state, and select action
                    h_t, z_t, h_normalized, activation_path, novelty = agent.perceive_and_update_state(cortex_id, state)
                    action, log_prob, _ = agent.select_action(actual_action_dim, activation_path)

                    # 2. Step environment
                    next_state, env_reward, terminated, truncated, _ = env.step(action)

                    # 3. Update STAG with the environmental reward
                    agent.update_stag(h_normalized, env_reward)

                    # 4. Calculate internal rewards
                    internal_reward = agent.get_internal_reward(damage_taken=0, novelty_signal=novelty)
                    total_reward = env_reward + internal_reward

                    # 5. Store experience for learning
                    agent.record_experience(h_t, z_t, activation_path, action, total_reward, next_state, terminated)

                    state = next_state
                    episode_reward += env_reward
                    steps_collected += 1
                    total_steps += 1

                    if terminated or truncated:
                        print(f"Episode finished. Reward: {episode_reward:.2f}, Total Steps: {total_steps}")
                        break

            # --- Sleep Phase: Train on Collected Data ---
            if len(agent.replay_buffer) > args.batch_size:
                print(f"\n--- Sleep Phase: Training ---")
                for i in range(10): # Train for 10 steps per sleep phase
                    # a. Train the world model
                    wm_stats = agent.train_world_model(cortex_id)
                    # b. Train the policy in imagination
                    ac_stats = agent.train_policy_in_imagination()

                    if i % 5 == 0: # Log every 5 train steps
                        print(f"  Train Step {i+1} | WM Loss: {wm_stats.get('wm_loss', -1):.4f} | AC Loss: {ac_stats.get('ac_loss', -1):.4f}")
            else:
                print("Not enough data to enter sleep phase, continuing collection.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        env.close()
        agent.save_state({"message": f"Training completed after {total_steps} steps."})
        print("Training finished and agent state saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ChimeraAgent using a local environment.")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Name of the gymnasium environment.")
    parser.add_argument("--agent_id", type=str, default=None, help="A unique ID for the agent. Defaults to 'agent-<env_name>'.")
    parser.add_argument("--total_steps", type=int, default=50000, help="Total number of steps to train for.")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for the agent.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards.")
    parser.add_argument("--force_new", action="store_true", help="Force creation of a new agent, ignoring saved state.")
    parser.add_argument("--no-stag", action="store_true", help="If set, STAG context will not be used in the actor-critic loss.")

    args = parser.parse_args()
    main(args)
