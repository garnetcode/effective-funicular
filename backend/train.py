import numpy as np
import os
import argparse
import asyncio
import uuid
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


async def main(args):
    """
    Main function to run the RL training loop with a Colosseum environment,
    implementing the Wake-Sleep cycle.
    """
    # --- Environment and Agent Setup ---
    print(f"Inspecting local environment for configuration: {args.env}")
    local_env = gym.make(args.env)
    cortex_configs, cortex_id = get_env_config(local_env)
    actual_action_dim = local_env.action_space.n
    local_env.close()

    agent_id = args.agent_id or f"agent-{args.env}"
    # Hyperparameters for the agent and training loop
    hyperparams = {
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'batch_size': args.batch_size,
        'imagine_horizon': args.horizon,
        'collect_interval': args.collect_interval,
        'train_steps': args.train_steps,
        'world_model_lr': 1e-3,
        'actor_critic_lr': 3e-4,
        'w_recon': 1.0, 'w_reward': 1.0, 'w_kl': 1.0,
        'w_policy': 1.0, 'w_critic': 0.5, 'w_entropy': 1e-4,
        'lambda': 0.95
    }
    agent = ChimeraAgent(
        agent_id=agent_id,
        max_obs_dim=256, # Assuming a max obs dim
        max_action_dim=256, # Assuming a max action dim
        cortex_configs=cortex_configs,
        load_from_storage=not args.force_new,
        hyperparams=hyperparams
    )
    print(f"Agent '{agent_id}' configured for '{args.env}'.")

    # --- Colosseum Connection ---
    player_token = args.token or str(uuid.uuid4())
    connector = ColosseumConnector(args.env, player_token, args.host, args.port)
    await connector.connect()

    # --- Main Training Loop (Wake-Sleep Cycle) ---
    total_steps = 0
    episode_num = 0
    try:
        while total_steps < args.total_steps:
            # --- Wake Phase: Collect Experience ---
            print(f"\n--- Wake Phase: Collecting experience for {args.collect_interval} steps ---")
            steps_collected = 0
            while steps_collected < args.collect_interval:
                episode_num += 1
                print(f"--- Episode {episode_num} ---")
                episode_reward = 0

                # Reset agent state at the start of each episode
                agent.hidden_state, agent.latent_state = agent.world_model.get_initial_state()
                agent.last_action = torch.tensor(0)

                # Wait for game start/turn
                msg = await connector.receive_message()
                if not msg or msg.get('type') == 'game.over': continue
                if msg.get('type') == 'match.start': msg = await connector.receive_message()

                state = msg.get('observation')
                terminated = False

                while not terminated:
                    # 1. Perceive, update state, and select action
                    h_t, z_t, h_normalized, activation_path, novelty = agent.perceive_and_update_state(cortex_id, np.array(state))
                    action, _ = agent.select_action(actual_action_dim, activation_path)

                    # 2. Step environment
                    await connector.send_action(int(action))
                    result_msg = await connector.receive_message()
                    if not result_msg or result_msg.get('type') != 'action.result': break

                    # 3. Process results and record experience
                    env_reward = result_msg.get('reward', 0)
                    damage = result_msg.get('damage', 0) # Assuming environment provides damage
                    next_state = result_msg.get('observation')
                    terminated = result_msg.get('terminated', False)

                    # 4. Update STAG with the environmental reward
                    agent.update_stag(h_normalized, env_reward)

                    # 5. Calculate internal rewards and add to env reward for policy training
                    internal_reward = agent.get_internal_reward(damage, novelty)
                    total_reward = env_reward + internal_reward

                    # 6. Store experience for learning
                    agent.record_experience(h_t, z_t, activation_path, action, total_reward, np.array(next_state), terminated)

                    episode_reward += reward
                    steps_collected += 1
                    total_steps += 1

                    if terminated:
                        print(f"Episode finished. Reward: {episode_reward:.2f}, Total Steps: {total_steps}")
                        break

                    # Wait for next turn
                    turn_msg = await connector.receive_message()
                    if not turn_msg or turn_msg.get('type') != 'game.turn': break
                    state = turn_msg.get('observation')

            # --- Sleep Phase: Train on Collected Data ---
            if len(agent.replay_buffer) > args.batch_size:
                print(f"\n--- Sleep Phase: Training for {args.train_steps} steps ---")
                for i in range(args.train_steps):
                    # a. Train the world model
                    wm_stats = agent.train_world_model(cortex_id)
                    # b. Train the policy in imagination
                    ac_stats = agent.train_policy_in_imagination()

                    if i % 50 == 0: # Log every 50 train steps
                        print(f"  Train Step {i+1}/{args.train_steps} | WM Loss: {wm_stats.get('wm_loss', -1):.4f} | AC Loss: {ac_stats.get('ac_loss', -1):.4f}")
            else:
                print("Not enough data to enter sleep phase, continuing collection.")

    except (websockets.exceptions.ConnectionClosedOK, KeyboardInterrupt):
        print("\nTraining interrupted or server closed connection.")
    finally:
        await connector.close()
        agent.save_state({"message": f"Training completed after {total_steps} steps."})
        print("Training finished and agent state saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ChimeraAgent using the Dreamer (Wake-Sleep) algorithm.")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Name of the Colosseum game environment.")
    parser.add_argument("--agent_id", type=str, default=None, help="A unique ID for the agent. Defaults to 'agent-<env_name>'.")
    parser.add_argument("--token", type=str, default=None, help="Player token for authentication. A new one is generated if not provided.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Hostname of the Colosseum server.")
    parser.add_argument("--port", type=int, default=8765, help="Port of the Colosseum server.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train for.")
    parser.add_argument("--dims", type=int, default=64, help="Dimensionality of the agent's internal brain space.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for the agent.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards.")
    parser.add_argument("--force_new", action="store_true", help="Force creation of a new agent, ignoring saved state.")

    args = parser.parse_args()
    asyncio.run(main(args))
