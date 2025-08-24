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
    env_names = [env.strip() for env in args.env_curriculum.split(',')]
    print(f"Starting training with curriculum: {env_names}")

    # --- Inspect all environments to determine max dimensions ---
    max_obs_dim = 0
    max_action_dim = 0
    cortex_configs = None
    cortex_id = None

    for name in env_names:
        print(f"Inspecting environment: {name}...")
        temp_env = gym.make(name)
        obs_space = temp_env.observation_space
        action_space = temp_env.action_space

        if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 1:
            max_obs_dim = max(max_obs_dim, obs_space.shape[0])
        else:
            raise NotImplementedError(f"Observation space type {type(obs_space)} not supported for {name}.")

        if isinstance(action_space, gym.spaces.Discrete):
            max_action_dim = max(max_action_dim, action_space.n)
        else:
            raise NotImplementedError(f"Action space type {type(action_space)} not supported for {name}.")

        if cortex_configs is None:
            # Use the first environment to determine the cortex config template
            cortex_configs, cortex_id = get_env_config(temp_env)

        temp_env.close()

    # Update cortex config to use the max observation dimension
    if cortex_configs and cortex_id in cortex_configs:
        cortex_configs[cortex_id]['params']['input_dim'] = max_obs_dim
    print(f"Max observation dimension: {max_obs_dim}, Max action dimension: {max_action_dim}")

    # --- Agent Setup ---
    agent_id = args.agent_id or f"agent-{env_names[0]}"
    hyperparams = {
        'learning_rate': args.lr, 'gamma': args.gamma, 'batch_size': args.batch_size,
        'imagine_horizon': 15, 'collect_interval': 100, 'train_steps': 10,
        'world_model_lr': 1e-3, 'actor_critic_lr': 3e-4,
        'w_recon': 1.0, 'w_reward': 1.0, 'w_kl': 1.0,
        'w_policy': 1.0, 'w_critic': 0.5, 'w_entropy': 1e-4, 'lambda': 0.95,
        'use_stag_in_ac_loss': not args.no_stag
    }
    agent = ChimeraAgent(
        agent_id=agent_id, max_obs_dim=max_obs_dim, max_action_dim=max_action_dim,
        cortex_configs=cortex_configs, load_from_storage=not args.force_new,
        hyperparams=hyperparams
    )
    print(f"Agent '{agent_id}' configured for curriculum.")

    # --- Main Training Loop (Curriculum) ---
    total_steps = 0
    episode_num = 0
    try:
        for i, env_name in enumerate(env_names):
            print(f"\n--- Starting Curriculum Stage {i+1}/{len(env_names)}: {env_name} ---")
            env = gym.make(env_name)
            actual_action_dim = env.action_space.n

            steps_in_current_env = 0
            while steps_in_current_env < args.steps_per_env and total_steps < args.total_steps:
                # --- Wake Phase: Collect Experience ---
                print(f"\n--- Wake Phase: Collecting experience ---")
                steps_collected = 0
                while steps_collected < 100 and steps_in_current_env < args.steps_per_env:
                    episode_num += 1
                    print(f"--- Episode {episode_num} ---")
                    episode_reward = 0
                    state, _ = env.reset()
                    agent.hidden_state, agent.latent_state = agent.world_model.get_initial_state()
                    agent.last_action = torch.tensor([0])
                    terminated, truncated = False, False

                    while not (terminated or truncated):
                        h_t, z_t, h_normalized, activation_path, novelty = agent.perceive_and_update_state(cortex_id, state)
                        action, _, _, _, _ = agent.select_action(actual_action_dim, activation_path)
                        next_state, env_reward, terminated, truncated, _ = env.step(action)
                        agent.update_stag(h_normalized, env_reward)
                        internal_reward = agent.get_internal_reward(damage_taken=0, novelty_signal=novelty)
                        total_reward = env_reward + internal_reward
                        agent.record_experience(h_t, z_t, activation_path, state, action, 0.0, total_reward, next_state, terminated)
                        state = next_state
                        episode_reward += env_reward
                        steps_collected += 1
                        total_steps += 1
                        steps_in_current_env += 1

                        if terminated or truncated:
                            print(f"Episode finished. Reward: {episode_reward:.2f}, Total Steps: {total_steps}")
                            break

                # --- Sleep Phase: Train on Collected Data ---
                if len(agent.replay_buffer) > args.batch_size:
                    print(f"\n--- Sleep Phase: Training ---")
                    for _ in range(10):
                        wm_stats = agent.train_world_model(cortex_id)
                        ac_stats = agent.train_policy_in_imagination()
                        print(f"  Train Step | WM Loss: {wm_stats.get('wm_loss', -1):.4f} | AC Loss: {ac_stats.get('ac_loss', -1):.4f}")
                else:
                    print("Not enough data to enter sleep phase, continuing collection.")

            env.close()
            print(f"--- Finished Curriculum Stage {i+1}/{len(env_names)}: {env_name} ---")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        agent.save_state({"message": f"Training completed after {total_steps} steps."})
        print("Training finished and agent state saved.")


def parse_args_and_run():
    """Parses command-line arguments and runs the main training function."""
    parser = argparse.ArgumentParser(description="Train a ChimeraAgent using a local environment or a curriculum.")
    parser.add_argument("--env-curriculum", type=str, default="CartPole-v1", help="A single env name or a comma-separated list of env names for curriculum learning.")
    parser.add_argument("--agent_id", type=str, default=None, help="A unique ID for the agent. Defaults to 'agent-<first_env_name>'.")
    parser.add_argument("--total_steps", type=int, default=50000, help="Total number of steps to train for across all environments.")
    parser.add_argument("--steps-per-env", type=int, default=50000, help="Number of steps to train on each environment in the curriculum.")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for the agent.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards.")
    parser.add_argument("--force_new", action="store_true", help="Force creation of a new agent, ignoring saved state.")
    parser.add_argument("--no-stag", action="store_true", help="If set, STAG context will not be used in the actor-critic loss.")

    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    parse_args_and_run()
