import numpy as np
import os
import argparse
import asyncio
import uuid
import torch
import random
from api.services.chimera_agent import ChimeraAgent
from api.services.cortex.factory import create_cortex_configs_from_observation_space

# NOTE: This script requires the `gymnasium` and `websockets` libraries.
# It was written assuming the libraries are installed. To run this, please ensure you have run:
# pip install -r requirements.txt

try:
    import gymnasium as gym
    import ale_py
except ImportError:
    print("FATAL: gymnasium library not found. Please install it with `pip install gymnasium[all]`")
    exit(1)


def seed_all(s):
    """Sets the seed for all random number generators."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    """
    Main function to run the RL training loop with a local environment,
    implementing the Wake-Sleep cycle and curriculum learning.
    """
    if args.seed is not None:
        print(f"Setting seed to {args.seed}")
        seed_all(args.seed)

    env_names = [env.strip() for env in args.env_curriculum.split(',')]
    print(f"Starting training with curriculum: {env_names}")

    # --- Inspect all environments to build a complete cortex configuration ---
    master_cortex_configs = {}
    max_action_dim = 0
    for name in env_names:
        print(f"Inspecting environment: {name}...")
        temp_env = gym.make(name)
        # Get the cortex config for this specific environment
        env_cortex_configs, _ = create_cortex_configs_from_observation_space(temp_env.observation_space)
        # Add it to our master dictionary
        master_cortex_configs.update(env_cortex_configs)

        # Determine the max action space size needed for the agent's action head
        if isinstance(temp_env.action_space, gym.spaces.Discrete):
            max_action_dim = max(max_action_dim, temp_env.action_space.n)
        else:
            raise NotImplementedError(f"Action space type {type(temp_env.action_space)} not supported for {name}.")
        temp_env.close()

    print(f"Master cortex configuration: {list(master_cortex_configs.keys())}")
    print(f"Max action dimension: {max_action_dim}")

    # --- Agent Setup ---
    # The agent is initialized once with all possible cortexes it might need.
    agent_id = args.agent_id or f"agent-{env_names[0]}"
    # Load hyperparameters from config.yaml
    import yaml
    with open("backend/config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            agent_config = config.get('agent_config', {})
            hyperparams = agent_config.get('hyperparams', {})
            embedding_dim = agent_config.get('embedding_dim', 256)
        except yaml.YAMLError as exc:
            print(exc)
            hyperparams = {}
            embedding_dim = 256

    # Override with command-line arguments if provided
    hyperparams['learning_rate'] = args.lr if args.lr is not None else hyperparams.get('learning_rate')
    hyperparams['gamma'] = args.gamma if args.gamma is not None else hyperparams.get('gamma')
    hyperparams['batch_size'] = args.batch_size if args.batch_size is not None else hyperparams.get('batch_size')
    hyperparams['use_stag_in_ac_loss'] = not args.no_stag

    agent = ChimeraAgent(
        agent_id=agent_id, embedding_dim=embedding_dim, max_action_dim=max_action_dim,
        cortex_configs=master_cortex_configs, load_from_storage=not args.force_new,
        hyperparams=hyperparams
    )
    # Set a placeholder goal
    goal = np.random.randn(agent.goal_dim)
    agent.set_goal(goal)
    print(f"Agent '{agent_id}' configured for curriculum and goal.")

    # --- Main Training Loop (Curriculum) ---
    total_steps = 0
    episode_num = 0
    try:
        for i, env_name in enumerate(env_names):
            print(f"\n--- Starting Curriculum Stage {i+1}/{len(env_names)}: {env_name} ---")
            agent.set_active_skill(env_name) # Set the active skill for the agent
            env = gym.make(env_name)
            actual_action_dim = env.action_space.n
            # Determine the correct cortex_id for this environment
            _, cortex_id = create_cortex_configs_from_observation_space(env.observation_space)

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
                    agent.hidden_state, agent.latent_state = agent.world_models[0].get_initial_state()
                    agent.last_action = torch.tensor([0], device=agent.device)
                    terminated, truncated = False, False

                    while not (terminated or truncated):
                        h_t, z_t, h_normalized, activation_path, novelty = agent.perceive_and_update_state(cortex_id, state)
                        action, log_prob, _, _, _ = agent.select_action(actual_action_dim, activation_path)
                        next_state, env_reward, terminated, truncated, _ = env.step(action)
                        agent.update_stag(h_normalized, env_reward)
                        internal_reward = agent.get_internal_reward(damage_taken=0, novelty_signal=novelty)
                        total_reward = env_reward + internal_reward
                        agent.record_experience(h_t, z_t, activation_path, state, action, log_prob, total_reward, next_state, terminated)
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
                        # The train method now encapsulates both WM and AC training
                        stats = agent.train(cortex_id)
                        if stats:
                            wm_loss = stats.get('wm_loss', -1)
                            ac_loss = stats.get('ac_loss', -1)
                            horizon = stats.get('horizon', -1)
                            entropy_coef = stats.get('entropy_coef', -1)
                            contrastive_loss = stats.get('contrastive_loss', -1)
                            reward_weights_norm = torch.norm(agent.reward_weights).item()
                            print(f"  Train Step {agent.train_steps} | WM Loss: {wm_loss:.4f} | AC Loss: {ac_loss:.4f} | Horizon: {horizon} | Entropy Coef: {entropy_coef:.4f} | Contrastive Loss: {contrastive_loss:.4f} | RW Norm: {reward_weights_norm:.4f}")
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
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    parse_args_and_run()
