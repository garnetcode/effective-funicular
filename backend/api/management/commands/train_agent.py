import numpy as np
import os
import torch
import yaml
import logging
import asyncio
from tqdm import tqdm
from django.core.management.base import BaseCommand
from api.services.chimera_agent import ChimeraAgent
from colosseum_connector import ColosseumConnector
from api.services.replay_buffer import Experience

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Train a ChimeraAgent against a Colosseum server.'

    def add_arguments(self, parser):
        parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")

    def _safe_format(self, value, format_spec):
        """Safely format a value, returning a placeholder if it's not a number."""
        if isinstance(value, (int, float)):
            return f"{value:{format_spec}}"
        return str(value)

    def handle(self, *args, **options):
        # We are calling an async method from a sync command.
        asyncio.run(self.a_handle(*args, **options))

    async def a_handle(self, *args, **options):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("training.log"),
                logging.StreamHandler()
            ]
        )

        # --- Load Configuration ---
        config_path = options['config']
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at: {config_path}")
            return

        logger.debug(f"Loaded configuration from {config_path}")

        # --- Environment & Agent Setup ---
        env_name = config['env_name']
        agent_id = config.get('agent_id', f"agent-{env_name}")
        agent_config = config.get('agent_config', {})
        history_config = config.get('agent_history', {})
        num_episodes = config.get('episodes_per_env', 100)

        # Load max dimensions from config
        max_obs_dim = agent_config.get('max_obs_dim', 2048)
        max_action_dim = agent_config.get('max_action_dim', 256)

        # --- Create a single, persistent agent ---
        agent = ChimeraAgent(
            agent_id=agent_id,
            max_obs_dim=max_obs_dim,
            max_action_dim=max_action_dim,
            latent_dim=agent_config.get('latent_dim', 64),
            hidden_dim=agent_config.get('hidden_dim', 128),
            cortex_configs={"vector_input": {"type": "DenseCortex", "params": {"input_dim": max_obs_dim}}},
            load_from_storage=not config.get('force_new_agent', False),
            hyperparams=agent_config.get('hyperparams', {}),
            history_config=history_config
        )
        logger.debug(f"Agent '{agent.agent_id}' created. Starting Colosseum training for {num_episodes} episodes in '{env_name}'...")

        # --- Create a single session for the entire training run ---
        connector = ColosseumConnector(env_name, agent_id)
        session_data = await connector.create_session()
        if not session_data:
            logger.error("Could not create Colosseum session. Exiting.")
            return

        if not await connector.connect_websocket():
            logger.error("WebSocket connection failed. Exiting.")
            return

        join_response = await connector.join_session()
        if not join_response:
            logger.error("Failed to join session. Exiting.")
            await connector.close()
            return

        # --- Training Loop ---
        total_rewards = []
        current_obs = np.array(join_response.get("observation"))

        # Get the actual action space size from the environment info
        try:
            actual_action_dim = join_response['action_space_shape']
            logger.debug(f"Environment action space size: {actual_action_dim}")
        except (KeyError, TypeError):
            logger.warning("Could not determine actual action space size from server. Defaulting to max.")
            actual_action_dim = max_action_dim


        burnin_steps = agent_config.get('hyperparams', {}).get('burnin_steps', 1000)
        logger.info(f"Starting burn-in phase for {burnin_steps} steps...")
        for _ in tqdm(range(burnin_steps), desc="Burn-in"):
            # Perceive the environment to get the latest state for the replay buffer
            _, _, _, activation_path, novelty = agent.perceive_and_update_state("vector_input", current_obs)

            # Take a random action
            action = np.random.randint(0, actual_action_dim)
            await connector.send_action(action)
            msg = await connector.receive_message()

            if not msg or msg.get("type") != "action.taken":
                if msg and msg.get("type") == "game.over":
                    reset_response = await connector.reset_environment()
                    if reset_response:
                        current_obs = np.array(reset_response.get("observation"))
                continue

            next_obs = np.array(msg.get("observation"))
            external_reward = msg.get("reward", 0)
            done = msg.get("done", False)

            # Store the random experience in the replay buffer
            # Use dummy values for log_prob as the action was random
            agent.record_experience(
                agent.hidden_state,
                agent.latent_state,
                activation_path,
                current_obs,
                action,
                torch.tensor(0.0), # Dummy log_prob
                external_reward, # Use external reward directly
                next_obs,
                done
            )

            current_obs = next_obs
            if done:
                reset_response = await connector.reset_environment()
                if reset_response:
                    current_obs = np.array(reset_response.get("observation"))


        try:
            with tqdm(total=num_episodes, desc="Training on Colosseum") as pbar:
                for episode in range(num_episodes):
                    episode_reward = 0
                    done = False

                    # --- Gameplay Loop for one episode ---
                    while not done:
                        # Perceive the environment and get agent's internal state and novelty
                        _, _, _, activation_path, novelty = agent.perceive_and_update_state("vector_input", current_obs)
                        action, log_prob, stag_context, decision_maker, epsilon = agent.select_action(actual_action_dim, activation_path)


                        await connector.send_action(action)
                        msg = await connector.receive_message()

                        if not msg:
                            logger.warning("Disconnection detected. Ending episode.")
                            done = True
                            continue

                        msg_type = msg.get("type")

                        if msg_type == "action.taken":
                            next_obs = np.array(msg.get("observation"))
                            external_reward = msg.get("reward")
                            done = msg.get("done")

                            # Get internal reward and update vitals
                            use_internal_reward = agent.hyperparams.get('use_internal_reward', True)
                            if use_internal_reward:
                                internal_reward = agent.get_internal_reward(damage_taken=0, novelty_signal=novelty)
                                total_reward = external_reward + internal_reward
                            else:
                                total_reward = external_reward

                            # Record experience with the combined, immediate reward
                            agent.record_experience(
                                agent.hidden_state,
                                agent.latent_state,
                                activation_path,
                                current_obs,
                                action,
                                log_prob,
                                total_reward,
                                next_obs,
                                done
                            )

                            current_obs = next_obs
                            episode_reward += external_reward  # Track original env reward for stats

                        elif msg_type == "game.over":
                            logger.debug(f"Game over message received. Final reward: {msg.get('final_reward')}")
                            done = True
                            # Final state might have a reward, let's process it.
                            # We assume the last 'action.taken' is sufficient for recording experience.
                            # A more robust implementation might handle a final reward here.
                            continue

                        elif msg_type == "viewer.joined":
                            logger.debug(f"Viewer joined message received. Ignoring.")
                            continue

                        else:
                            logger.warning(f"Unexpected message type received: {msg_type}")

                    # --- Post-Episode: Train the agent ---
                    logger.debug(f"Episode {episode + 1} finished. Reward: {episode_reward:.2f}. Training...")
                    train_stats = agent.train(cortex_id="vector_input")
                    if (episode + 1) % 1000 == 0:
                        agent.save_state(version_info=train_stats)
                    total_rewards.append(episode_reward)

                    avg_reward = np.mean(total_rewards[-100:])
                    pbar.set_postfix({
                        "Reward": f"{episode_reward:.2f}",
                        "Avg Rwd": f"{avg_reward:.2f}",
                        "Epsilon": f"{epsilon:.3f}",
                        "Energy": f"{agent.energy:.1f}",
                        "Integrity": f"{agent.integrity:.1f}",
                        "Decision": decision_maker,
                        "Action Prob": f"{torch.exp(log_prob).item():.3f}",
                        "Policy Loss": self._safe_format(train_stats.get('policy_loss'), '.4f')
                    })
                    pbar.update(1)

                    # --- Reset for next episode ---
                    if episode < num_episodes - 1:
                        reset_response = await connector.reset_environment()
                        if reset_response:
                            current_obs = np.array(reset_response.get("observation"))
                        else:
                            logger.error("Failed to reset environment, stopping training.")
                            break

        finally:
            # --- Final cleanup ---
            await connector.close()
            logger.debug("Training finished.")
        logger.info(f"Final agent state saved to {agent.history_manager.storage_dir}")
