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
    Main function to run the RL training loop with a Colosseum environment.
    """
    # --- Local Env for Config ---
    print(f"Inspecting local environment for configuration: {args.env}")
    local_env = gym.make(args.env)
    cortex_configs, cortex_id = get_env_config(local_env)
    local_env.close()

    # --- Agent Initialization ---
    agent_id = args.agent_id or f"agent-{args.env}"
    n_actions = 256  # Hardcoded as per user specification

    agent = ChimeraAgent(
        agent_id=agent_id,
        dimensions=args.dims,
        n_actions=n_actions,
        cortex_configs=cortex_configs,
        load_from_storage=not args.force_new,
        hyperparams={'learning_rate': args.lr, 'gamma': args.gamma}
    )

    print(f"Agent '{agent_id}' configured for '{args.env}'.")
    print(f"Agent brain dimensions: {args.dims}, Actions: {n_actions}")

    # --- Colosseum Connection ---
    player_token = args.token or str(uuid.uuid4())
    connector = ColosseumConnector(args.env, player_token, args.host, args.port)
    await connector.connect()

    # --- Training Loop ---
    total_rewards = []
    try:
        for episode in range(args.episodes):
            print(f"--- Episode {episode + 1}/{args.episodes} ---")
            episode_reward = 0
            terminated = False

            # Wait for the game to start and for our turn
            while True:
                msg = await connector.receive_message()
                if not msg: break

                if msg.get('type') == 'match.start':
                    print("Match has started!")
                elif msg.get('type') == 'game.turn':
                    print("Agent's turn.")
                    state = msg.get('observation')
                    break # Ready to act
                elif msg.get('type') == 'game.over':
                    print("Game over before agent's turn. Starting new episode.")
                    terminated = True
                    break

            if terminated:
                continue

            # Main interaction loop for the episode
            while not terminated:
                # 1. Perceive the state and update the hidden state
                hidden_state = agent.perceive_and_update_state(cortex_id, np.array(state))

                # 2. Select an action
                action, log_prob, stag_context = agent.select_action(local_env.action_space.n)

                # 3. Take action in the environment
                await connector.send_action(int(action))

                # 4. Wait for the result of the action
                result_msg = await connector.receive_message()
                if not result_msg or result_msg.get('type') != 'action.result':
                    print(f"Unexpected message or connection closed: {result_msg}")
                    break

                reward = result_msg.get('reward', 0)
                print(f"Received reward: {reward}")
                is_terminated = result_msg.get('terminated', False)
                next_state = result_msg.get('observation')

                # 5. Record the experience
                agent.record_experience(hidden_state, stag_context, np.array(state), action, log_prob, reward, np.array(next_state), is_terminated)
                episode_reward += reward

                if is_terminated:
                    print(f"Episode finished. Total reward: {episode_reward:.2f}")
                    break

                # 6. Wait for the next turn
                turn_msg = await connector.receive_message()
                if not turn_msg or turn_msg.get('type') != 'game.turn':
                    print(f"Unexpected message or connection closed: {turn_msg}")
                    break
                state = turn_msg.get('observation')

            # End of episode: Train the agent
            agent.train()
            total_rewards.append(episode_reward)

            if episode % 10 == 0 and episode > 0:
                avg_reward = np.mean(total_rewards[-100:])
                print(f"Episode {episode+1} | Avg Reward (last 100): {avg_reward:.2f}")

    except websockets.exceptions.ConnectionClosedOK:
        print("Colosseum server closed the connection gracefully.")
    finally:
        await connector.close()
        print("Training finished.")
        print(f"Final agent state saved to {agent.storage_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ChimeraAgent with a Colosseum environment.")
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
