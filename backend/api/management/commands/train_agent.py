import numpy as np
import os
from django.core.management.base import BaseCommand
from api.services.chimera_agent import ChimeraAgent

try:
    import gymnasium as gym
except ImportError:
    # Set gym to None if it's not installed, so the command can still be loaded by Django
    gym = None

def get_env_config(env):
    """Inspects a gymnasium environment to determine agent configuration."""
    obs_space = env.observation_space
    if not isinstance(obs_space, gym.spaces.Box) or len(obs_space.shape) != 1:
        raise NotImplementedError("Only 1D Box observation spaces are supported.")

    act_space = env.action_space
    if not isinstance(act_space, gym.spaces.Discrete):
        raise NotImplementedError("Only Discrete action spaces are supported.")

    input_dim = obs_space.shape[0]
    cortex_configs = {"vector_input": {"type": "DenseCortex", "params": {"input_dim": input_dim}}}
    n_actions = act_space.n
    return cortex_configs, "vector_input", n_actions

class Command(BaseCommand):
    help = 'Train a ChimeraAgent in a Gymnasium environment.'

    def add_arguments(self, parser):
        parser.add_argument("--env", type=str, default="CartPole-v1", help="Name of the gymnasium environment.")
        parser.add_argument("--agent_id", type=str, default=None, help="A unique ID for the agent. Defaults to 'agent-<env_name>'.")
        parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train for.")
        parser.add_argument("--dims", type=int, default=64, help="Dimensionality of the agent's internal brain space.")
        parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for the agent.")
        parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards.")
        parser.add_argument("--force_new", action="store_true", help="Force creation of a new agent, ignoring saved state.")

    def handle(self, *args, **options):
        if not gym:
            self.stderr.write(self.style.ERROR("FATAL: gymnasium library not found. Please install it with `pip install gymnasium`"))
            return

        env_name = options['env']
        self.stdout.write(f"Initializing environment: {env_name}")
        env = gym.make(env_name)

        cortex_configs, cortex_id, n_actions = get_env_config(env)

        agent_id = options['agent_id'] or f"agent-{env_name}"

        agent = ChimeraAgent(
            agent_id=agent_id,
            dimensions=options['dims'],
            n_actions=n_actions,
            cortex_configs=cortex_configs,
            load_from_storage=not options['force_new'],
            hyperparams={'learning_rate': options['lr'], 'gamma': options['gamma']}
        )

        self.stdout.write(self.style.SUCCESS(f"Starting training for agent '{agent_id}' in '{env_name}'..."))
        self.stdout.write(f"Agent brain dimensions: {options['dims']}, Actions: {n_actions}")

        total_rewards = []
        for episode in range(options['episodes']):
            state, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0

            while not (terminated or truncated):
                # --- Homeostasis Integration ---
                # 1. Get vitals before the action
                old_energy = agent.energy
                old_integrity = agent.integrity

                # 2. Perceive state and select action
                state_embedding = agent.perceive(cortex_id, state)
                action, log_prob, internal_state = agent.select_action(state_embedding)

                # 3. Step the environment
                next_state, external_reward, terminated, truncated, info = env.step(action)

                # 4. Update agent vitals
                agent.energy -= agent.metabolic_cost # Apply metabolic cost
                agent.energy += info.get('energy_change', 0.0)
                agent.integrity += info.get('integrity_change', 0.0)
                agent.energy = min(agent.energy, agent.max_energy) # Clamp energy

                # 5. Calculate homeostatic reward
                energy_reward = agent.energy - old_energy
                integrity_reward = agent.integrity - old_integrity
                homeostatic_reward = energy_reward + integrity_reward

                # 6. Calculate total reward
                total_reward = external_reward + homeostatic_reward

                # 7. Record experience with the total reward
                agent.record_experience(internal_state, action, log_prob, total_reward)

                state = next_state
                episode_reward += external_reward # Track external reward for reporting

            agent.train()
            total_rewards.append(episode_reward)

            if episode % 10 == 0:
                avg_reward = np.mean(total_rewards[-100:])
                self.stdout.write(f"Episode {episode}/{options['episodes']} | Total Reward: {episode_reward:.2f} | Avg Reward (last 100): {avg_reward:.2f}")

        env.close()
        self.stdout.write(self.style.SUCCESS("Training finished."))
        self.stdout.write(f"Final agent state saved to {agent.storage_path}")
