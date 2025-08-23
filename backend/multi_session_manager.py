import asyncio
import uuid
import logging
import numpy as np
import torch
from colosseum_connector import ColosseumConnector
from api.services.chimera_agent import ChimeraAgent

logger = logging.getLogger(__name__)

class MultiSessionManager:
    """
    Orchestrates multiple ChimeraAgent sessions concurrently.
    """
    def __init__(self, agent_config, history_config, env_list, num_episodes):
        self.agent_config = agent_config
        self.history_config = history_config
        self.env_list = env_list
        self.num_episodes = num_episodes
        self.agents = {}

    async def run_agent_session(self, env_id):
        """
        Manages the lifecycle of a single agent session, from creation to completion,
        running for a specified number of episodes.
        """
        agent_tag = f"chimera-agent-{env_id}-{uuid.uuid4()}"
        logger.info(f"[{agent_tag}] Starting session for {self.num_episodes} episodes in environment: {env_id}")

        # 1. Create a connector for this session
        connector = ColosseumConnector(env_id, agent_tag)

        # 2. Create the Colosseum session
        session_data = await connector.create_session()
        if not session_data:
            logger.error(f"[{agent_tag}] Could not create Colosseum session. Exiting.")
            return

        # 3. Create a ChimeraAgent instance for this session
        obs_dim = len(session_data.get("observation", []))
        action_dim = 4 if "Lunar" in env_id else 2

        agent = ChimeraAgent(
            agent_id=agent_tag,
            obs_dim=obs_dim,
            action_dim=action_dim,
            history_config=self.history_config,
            **self.agent_config
        )
        self.agents[agent_tag] = agent

        # 4. Connect to the WebSocket and join the session
        if not await connector.connect_websocket():
            logger.error(f"[{agent_tag}] WebSocket connection failed. Exiting.")
            return

        join_response = await connector.join_session()
        if not join_response:
            logger.error(f"[{agent_tag}] Failed to join session over WebSocket. Exiting.")
            await connector.close()
            return

        # 6. Main training loop
        try:
            current_obs = np.array(join_response.get("observation"))
            try:
                actual_action_dim = join_response['action_space_shape']
                logger.info(f"[{agent_tag}] Environment action space size: {actual_action_dim}")
            except (KeyError, TypeError):
                logger.warning(f"[{agent_tag}] Could not determine action space size. Defaulting to {action_dim}.")
                actual_action_dim = action_dim

            for episode in range(self.num_episodes):
                done = False
                episode_reward = 0

                while not done:
                    agent.perceive_and_update_state("test_cortex", current_obs)
                    action, log_prob, stag_context = agent.select_action(actual_action_dim)

                    await connector.send_action(action)
                    msg = await connector.receive_message()

                    if not msg:
                        logger.warning(f"[{agent_tag}] Disconnection detected. Ending episode.")
                        done = True
                        continue

                    msg_type = msg.get("type")

                    if msg_type == "action.taken":
                        next_obs = np.array(msg.get("observation"))
                        reward = msg.get("reward")
                        done = msg.get("done")

                        total_reward = reward
                        agent.record_experience(agent.hidden_state, stag_context, current_obs, action, log_prob, total_reward, next_obs, done)
                        agent.train()

                        current_obs = next_obs
                        episode_reward += reward

                    elif msg_type == "game.over":
                        logger.info(f"[{agent_tag}] Game over message received. Final reward: {msg.get('final_reward')}")
                        done = True
                        continue

                    else:
                        logger.warning(f"[{agent_tag}] Unexpected message type received: {msg_type}")

                logger.info(f"[{agent_tag}] Episode {episode + 1}/{self.num_episodes} gameplay loop finished. Reward: {episode_reward:.2f}. Resetting environment...")

                if episode < self.num_episodes - 1:
                    reset_response = await connector.reset_environment()
                    if reset_response:
                        current_obs = np.array(reset_response.get("observation"))
                    else:
                        logger.error(f"[{agent_tag}] Failed to reset environment. Stopping.")
                        break

        except Exception as e:
            logger.error(f"[{agent_tag}] An error occurred during the session: {e}", exc_info=True)
        finally:
            await connector.close()
            logger.info(f"[{agent_tag}] Session closed.")

    async def start(self):
        """
        Starts the concurrent training sessions for all environments.
        """
        tasks = [self.run_agent_session(env_id) for env_id in self.env_list]
        await asyncio.gather(*tasks)
