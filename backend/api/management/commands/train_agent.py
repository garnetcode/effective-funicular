import numpy as np
import os
import time
import torch
import yaml
import logging
import asyncio
import pprint
from tqdm import tqdm
from django.core.management.base import BaseCommand
from api.services.chimera_agent import ChimeraAgent
from colosseum_connector import ColosseumConnector
from api.services.replay_buffer import Experience
from api.signals import agent_data_signal
from api.services.cortex.factory import create_cortex_configs_from_observation_space
import gymnasium as gym # Needed for space deserialization

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Train a ChimeraAgent against a Colosseum server.'

    def add_arguments(self, parser):
        parser.add_argument("--config", type=str, default="configs/phase_0.yaml", help="Path to the main configuration file.")

    def _safe_format(self, value, format_spec):
        """Safely format a value, returning a placeholder if it's not a number."""
        if isinstance(value, (int, float)):
            return f"{value:{format_spec}}"
        return str(value)

    def _deep_merge(self, source, destination):
        """Recursively merge source dict into destination dict."""
        for key, value in source.items():
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                self._deep_merge(value, node)
            else:
                destination[key] = value
        return destination

    def _load_config(self, config_path):
        """Loads a YAML config file and handles imports."""
        try:
            with open(config_path, 'r') as f:
                main_config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at: {config_path}")
            return None

        # Handle imports
        if 'imports' in main_config:
            base_config = {}
            config_dir = os.path.dirname(config_path)
            for import_file in main_config['imports']:
                import_path = os.path.join(config_dir, import_file)
                imported_config = self._load_config(import_path)
                if imported_config:
                    base_config = self._deep_merge(base_config, imported_config)

            # Merge main config over the base
            config = self._deep_merge(base_config, main_config)
        else:
            config = main_config

        return config

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
        config_path = os.path.join('backend', options['config'])
        config = self._load_config(config_path)
        if not config:
            return

        logger.debug(f"Loaded and merged configuration from {config_path}")

        # --- Set Seed for Reproducibility ---
        seed = config.get('training', {}).get('seed')
        if seed is not None:
            logger.info(f"Setting random seed to {seed}")
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # --- Environment & Agent Setup ---
        env_name = config['env_name']
        agent_id = config.get('agent_id', f"agent-{env_name}")
        agent_config = config.get('agent_config', {})
        history_config = config.get('agent_history', {})
        num_episodes = config.get('episodes_per_env', 100)

        # --- Connect to Colosseum and get environment specs ---
        connector = ColosseumConnector(env_name, agent_id)
        session_data = await connector.create_session()
        if not session_data:
            logger.error("Could not create Colosseum session. Exiting.")
            return

        logger.info(f"Received session data: {pprint.pformat(session_data)}")

        # --- Dynamically configure agent based on environment specs from the session data ---
        try:
            # Reconstruct the observation space from the server's response
            obs_space_info = session_data['environment']['observation_space']
            observation_space = gym.spaces.Box(
                low=np.array(obs_space_info['low']),
                high=np.array(obs_space_info['high']),
                shape=obs_space_info['shape'],
                dtype=np.dtype(obs_space_info['dtype'])
            )
            actual_action_dim = session_data['environment']['action_space']['n']

            cortex_configs, cortex_id = create_cortex_configs_from_observation_space(observation_space)
            logger.info(f"Dynamically configured cortex: '{cortex_id}' for observation space {observation_space}")

        except (KeyError, TypeError) as e:
            logger.error(f"Could not determine environment specs from server response: {e}. Exiting.")
            return

        # AGENT_FIX: Dynamically set goal_dim to match the observation space dimension.
        # This is the core of the fix for HER, ensuring that goals and observations
        # have compatible shapes.
        obs_dim = np.prod(observation_space.shape)
        if 'hyperparams' not in agent_config:
            agent_config['hyperparams'] = {}
        agent_config['hyperparams']['goal_dim'] = obs_dim
        logger.info(f"Dynamically setting goal_dim to observation space dimension: {obs_dim}")


        # Load agent architecture config
        embedding_dim = agent_config.get('embedding_dim', 256)
        max_action_dim = agent_config.get('max_action_dim', 256)

        # --- Create a single, persistent agent ---
        agent = ChimeraAgent(
            agent_id=agent_id,
            embedding_dim=embedding_dim,
            max_action_dim=max_action_dim,
            latent_dim=agent_config.get('latent_dim', 64),
            hidden_dim=agent_config.get('hidden_dim', 128),
            cortex_configs=cortex_configs,
            load_from_storage=not config.get('force_new_agent', False),
            hyperparams=agent_config.get('hyperparams', {}),
            history_config=history_config
        )
        agent.set_active_skill(env_name) # Set the active skill for the agent
        logger.debug(f"Agent '{agent.agent_id}' created. Starting Colosseum training for {num_episodes} episodes in '{env_name}'...")

        # --- Connect to WebSocket and start training ---
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
        training_config = config.get('training', {})

        # Restore training loop state if available
        if agent.loaded_snapshot_data:
            snapshot_data = agent.loaded_snapshot_data
            total_steps = snapshot_data.get('total_steps', 0)
            best_avg_reward = snapshot_data.get('best_avg_reward', -float('inf'))
            last_checkpoint_step = snapshot_data.get('last_checkpoint_step', 0)
            last_refresh_step = snapshot_data.get('last_refresh_step', 0)
            logger.info(f"Restored training loop state. Total steps: {total_steps}, Best Avg Reward: {best_avg_reward:.2f}")
        else:
            total_steps = 0
            best_avg_reward = -float('inf')
            last_checkpoint_step = 0
            last_refresh_step = 0

        hyperparams = agent_config.get('hyperparams', {})
        dr_config = hyperparams.get('domain_randomization', {})
        obs_noise_level = dr_config.get('obs_noise_level', 0.0)
        action_delay_prob = dr_config.get('action_delay_prob', 0.0)
        last_sent_action = 0
        checkpoint_interval = training_config.get('checkpoint_every_n_steps', 50000)
        save_on_best = training_config.get('save_on_best_return', True)

        refresh_interval = hyperparams.get('on_policy_refresh_interval', 20000)
        refresh_duration = hyperparams.get('on_policy_refresh_duration', 2000)
        on_policy_epsilon = hyperparams.get('on_policy_epsilon', 0.1)

        current_obs = np.array(session_data.get("observation"))
        burnin_steps = agent_config.get('hyperparams', {}).get('burnin_steps', 1000)
        logger.info(f"Starting burn-in phase for {burnin_steps} steps...")
        for _ in tqdm(range(burnin_steps), desc="Burn-in"):
            # Perceive the environment to get the latest state for the replay buffer
            _, _, _, activation_path, novelty = agent.perceive_and_update_state(cortex_id, current_obs)

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
            max_training_steps = config.get('max_training_steps', 2000000)
            with tqdm(total=max_training_steps, initial=total_steps, desc="Training on Colosseum") as pbar:
                while total_steps < max_training_steps:
                    episode_reward = 0
                    done = False
                    episode_start_step = total_steps

                    # --- Gameplay Loop for one episode ---
                    while not done:
                        # 1. Domain Randomization (Observation)
                        obs_to_perceive = current_obs
                        if obs_noise_level > 0:
                            noise = np.random.normal(0, obs_noise_level, size=current_obs.shape)
                            obs_to_perceive = current_obs + noise

                        # 2. Perceive and Select Action
                        _, _, _, activation_path, novelty = agent.perceive_and_update_state(cortex_id, obs_to_perceive)
                        action, log_prob, stag_context, decision_maker, epsilon, action_time = agent.select_action(actual_action_dim, activation_path)

                        # 3. Domain Randomization (Action)
                        if random.random() < action_delay_prob:
                            action_to_send = last_sent_action
                        else:
                            action_to_send = action
                        last_sent_action = action_to_send

                        # 4. Step the environment
                        await connector.send_action(action_to_send)
                        msg = await connector.receive_message()

                        pbar.update(1) # Increment progress bar for each environment step

                        if not msg:
                            logger.warning("Disconnection detected. Ending episode.")
                            done = True
                            continue

                        msg_type = msg.get("type")

                        if msg_type == "action.taken":
                            next_obs = np.array(msg.get("observation"))
                            external_reward = msg.get("reward")
                            done = msg.get("done")

                            # 3. Process reward and record experience
                            use_internal_reward = agent.hyperparams.get('use_internal_reward', False)
                            if use_internal_reward:
                                internal_reward = agent.get_internal_reward(damage_taken=0, novelty_signal=novelty)

                                # Adaptive Reward Mixing
                                reward_mix_params = agent.hyperparams.get('reward_mixing', {})
                                schedule_steps = reward_mix_params.get('schedule_steps', 1)
                                start_weight = reward_mix_params.get('intrinsic_start_weight', 0.0)
                                end_weight = reward_mix_params.get('intrinsic_end_weight', 0.0)

                                progress = min(1.0, total_steps / schedule_steps)
                                intrinsic_weight = start_weight + progress * (end_weight - start_weight)
                                extrinsic_weight = 1.0 - intrinsic_weight

                                total_reward = (extrinsic_weight * external_reward) + (intrinsic_weight * internal_reward)
                            else:
                                total_reward = external_reward

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
                            episode_reward += external_reward
                            total_steps += 1

                        elif msg_type == "game.over":
                            logger.debug(f"Game over message received. Final reward: {msg.get('final_reward')}")
                            done = True
                            continue
                        elif msg_type == "viewer.joined":
                            logger.debug(f"Viewer joined message received. Ignoring.")
                            continue
                        else:
                            logger.warning(f"Unexpected message type received: {msg_type}")

                    # --- Post-Episode Logic ---
                    episode_steps = total_steps - episode_start_step
                    logger.debug(f"Episode finished in {episode_steps} steps. Reward: {episode_reward:.2f}. Training...")

                    # --- Training, Checkpointing, and Logging ---
                    phase = training_config.get('phase', 1)
                    if phase == 1:
                        stats, _, _ = agent.train_world_model(cortex_id=cortex_id)
                        train_stats = stats or {}
                    else:
                        train_stats = agent.train(cortex_id=cortex_id)

                    total_rewards.append(episode_reward)
                    avg_reward = np.mean(total_rewards[-100:]) if total_rewards else 0

                    # Checkpointing, etc.
                    training_loop_state = {'total_steps': total_steps, 'best_avg_reward': best_avg_reward, 'last_checkpoint_step': last_checkpoint_step, 'last_refresh_step': last_refresh_step}
                    if save_on_best and avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        logger.info(f"New best average reward: {best_avg_reward:.2f}. Saving best model...")
                        agent.save_state(version_info={**train_stats, **training_loop_state})
                    if total_steps >= last_checkpoint_step + checkpoint_interval:
                        last_checkpoint_step = total_steps
                        logger.info(f"Reached {total_steps} steps. Saving periodic checkpoint...")
                        agent.save_state(version_info={**train_stats, **training_loop_state})

                    # Update progress bar
                    postfix_metrics = {
                        "Reward": f"{episode_reward:.2f}",
                        "Avg Rwd": f"{avg_reward:.2f}",
                        "Epsilon": f"{epsilon:.3f}",
                        "Energy": f"{agent.energy:.1f}",
                        "Integrity": f"{agent.integrity:.1f}",
                        "Decision": decision_maker,
                        "Action Time": f"{action_time:.4f}s",
                        # World Model losses
                        "WM_Total": self._safe_format(train_stats.get('wm_loss_total'), '.4f'),
                        "W_Recon": self._safe_format(train_stats.get('wm_loss_recon'), '.4f'),
                        "W_KL": self._safe_format(train_stats.get('wm_loss_kl'), '.4f'),
                        # Policy losses
                        "AC_Total": self._safe_format(train_stats.get('ac_loss'), '.4f'),
                        "Policy": self._safe_format(train_stats.get('policy_loss'), '.4f'),
                        "Critic": self._safe_format(train_stats.get('critic_loss'), '.4f'),
                        "Entropy": self._safe_format(train_stats.get('entropy'), '.4f'),
                        "BC_Loss": self._safe_format(train_stats.get('bc_loss'), '.4f'),
                    }
                    pbar.set_postfix(postfix_metrics)

                    # Environment reset is handled by the game server on 'done'
                    if not await connector.is_connected():
                        logger.error("Lost connection to server. Stopping training.")
                        break

        finally:
            # --- Final cleanup ---
            await connector.close()
            logger.debug("Training finished.")
        logger.info(f"Final agent state saved to {agent.history_manager.storage_dir}")
