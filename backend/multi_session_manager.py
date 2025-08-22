import asyncio
import uuid
import logging
import numpy as np
import torch
from .colosseum_connector import ColosseumConnector
from .api.services.chimera_agent import ChimeraAgent

logger = logging.getLogger(__name__)

class MultiSessionManager:
    """
    Orchestrates multiple ChimeraAgent sessions concurrently.
    """
    def __init__(self, agent_config, env_list):
        self.agent_config = agent_config
        self.env_list = env_list
        self.agents = {}

    async def run_agent_session(self, env_id):
        """
        Manages the lifecycle of a single agent session, from creation to completion.
        """
        agent_tag = f"chimera-agent-{env_id}-{uuid.uuid4()}"
        logger.info(f"[{agent_tag}] Starting session for environment: {env_id}")

        # 1. Create a connector for this session
        connector = ColosseumConnector(env_id, agent_tag)

        # 2. Create the Colosseum session
        session_data = await connector.create_session()
        if not session_data:
            logger.error(f"[{agent_tag}] Could not create Colosseum session. Exiting.")
            return

        # 3. Create a ChimeraAgent instance for this session
        # We need to get obs_dim and action_dim from the environment spec
        # This is a simplification; a real implementation would query this from the server
        # or have it in a config file.
        obs_dim = len(session_data.get("observation", []))
        # This is a HACK: we need a way to know the action space size.
        action_dim = 4 if "Lunar" in env_id else 2 # Default for CartPole

        agent = ChimeraAgent(
            agent_id=agent_tag,
            obs_dim=obs_dim,
            action_dim=action_dim,
            **self.agent_config
        )
        self.agents[agent_tag] = agent

        # 4. Connect to the WebSocket
        if not await connector.connect_websocket():
            logger.error(f"[{agent_tag}] WebSocket connection failed. Exiting.")
            return

        # 5. Join the session over WebSocket
        join_response = await connector.join_session()
        if not join_response:
            logger.error(f"[{agent_tag}] Failed to join session over WebSocket. Exiting.")
            await connector.close()
            return

        # 6. Main real-time gameplay loop
        try:
            current_obs = np.array(join_response.get("observation"))
            done = False

            while not done:
                # Agent perceives, acts, and learns online
                agent.perceive_and_update_state("test_cortex", current_obs) # test_cortex is a placeholder
                action, _ = agent.select_action()

                await connector.send_action(action)

                msg = await connector.receive_message()
                if not msg or msg.get("type") != "action.taken":
                    logger.warning(f"[{agent_tag}] Unexpected message or disconnection: {msg}")
                    break

                # Extract results
                next_obs = np.array(msg.get("observation"))
                reward = msg.get("reward")
                done = msg.get("done")

                # Homeostatic reward calculation
                old_energy = agent.energy
                agent.energy -= agent.metabolic_cost
                # NOTE: The Colosseum API v4 spec doesn't show energy/integrity change in info dict.
                # We are only using metabolic cost for now.
                homeostatic_reward = agent.energy - old_energy
                total_reward = reward + homeostatic_reward

                # Record and train
                # The log_prob is returned by select_action and needed for the policy loss
                agent.record_experience(current_obs, action, log_prob, total_reward, next_obs, done)
                agent.train()

                current_obs = next_obs

            logger.info(f"[{agent_tag}] Session finished. Final reward: {msg.get('total_reward', 0)}")

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
