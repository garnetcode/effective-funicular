# Implements the ChimeraAgent, the core orchestrator for the cognitive architecture.
# This class encapsulates the agent's "brain" (cognitive layers), its sensory
# cortexes, and its action-selection mechanism. It is responsible for managing
# the flow of information between the layers as specified in the Project Chimera doc.

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

logger = logging.getLogger(__name__)

class NumpyJSONEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

from .world_model_core import WorldModel
from .stag_framework import STAG_Framework
from .state_history_manager import StateHistoryManager
from .replay_buffer import ReplayBuffer, Experience
from . import cortex as cortex_modules
from .action.modules import ActionHead
from .action.generation_head import TextGenerationHead

class StagContextProcessor(nn.Module):
    """
    Processes the STAG's activation path to create a rich context vector C_t.
    This version uses a simple linear layer over the concatenated node weights.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.processor = nn.Linear(input_dim, output_dim)

    def forward(self, activation_path_weights):
        # activation_path_weights is a list of weight vectors.
        # We concatenate them to form a single input vector.
        if not activation_path_weights:
            return torch.zeros(1, self.processor.out_features) # Return a zero vector if path is empty

        concatenated_weights = torch.cat(activation_path_weights, dim=0)
        return self.processor(concatenated_weights.unsqueeze(0))


def _initialize_cortexes(configs, output_dim):
    """
    Initializes all cortex modules based on the provided configurations.
    It dynamically loads the class from the cortex package.
    """
    cortexes = {}
    if configs is None: return cortexes
    for cortex_id, config in configs.items():
        class_name = config['type']
        params = config.get('params', {})
        try:
            CortexClass = getattr(cortex_modules, class_name)

            # Pass parameters based on cortex type
            if class_name == "DenseCortex":
                cortexes[cortex_id] = CortexClass(input_dim=params['input_dim'], output_dim=output_dim)
            elif class_name == "LanguageCortex":
                cortexes[cortex_id] = CortexClass(
                    model_path_or_id=params['model_id'],
                    output_dim=output_dim,
                    api_base=params.get('api_base'),
                    embedding_dim=params.get('embedding_dim')
                )
            else: # For TextCortex, VisionCortex etc.
                cortexes[cortex_id] = CortexClass(output_dim=output_dim)
        except (AttributeError, ImportError) as e:
            print(f"Warning: Could not initialize cortex '{cortex_id}' of type '{class_name}': {e}")
    return cortexes

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class ChimeraAgent:
    def __init__(self, agent_id, max_obs_dim, max_action_dim, latent_dim=32, hidden_dim=200, cortex_configs=None, load_from_storage=True, hyperparams=None, history_config=None, **kwargs):
        self.agent_id = agent_id
        self.max_obs_dim = max_obs_dim
        self.max_action_dim = max_action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Consolidate hyperparameters from explicit arg and kwargs
        self.hyperparams = hyperparams or {}
        self.hyperparams.update(kwargs)

        history_config = history_config or {}
        self.history_manager = StateHistoryManager(agent_id, **history_config)
        self.learning_rate = self.hyperparams.get('learning_rate', 0.0001)
        self.gamma = self.hyperparams.get('gamma', 0.99)
        self.stag_expansion_threshold = self.hyperparams.get('stag_expansion_threshold', 0.1)
        self.max_grad_norm = self.hyperparams.get('max_grad_norm', 1.0)
        # Max length of the STAG activation path for the context processor
        self.max_stag_path_length = self.hyperparams.get('max_stag_path_length', 10)
        self.stag_context_dim = self.hyperparams.get('stag_context_dim', 128)

        # --- Add Homeostatic Vitals ---
        self.max_energy = 100.0
        self.energy = self.max_energy
        self.integrity = 100.0
        self.max_integrity = 100.0
        self.metabolic_cost = 0.1

        # --- Initialize Agent State and Components ---
        self.steps_done = 0
        # h_t and z_t for the RSSM
        self.hidden_state = torch.zeros(1, self.hidden_dim)
        self.latent_state = torch.zeros(1, self.latent_dim)
        self.last_action = torch.tensor([0])
        self.cortex_configs = cortex_configs or {}

        # Language Model setup (remains the same)
        lm_config = self.hyperparams.get('language_model', {})
        self.language_model_enabled = lm_config.get('enabled', False)
        embedding_model_id = None
        generation_model_id = None
        if self.language_model_enabled:
            embedding_model_id = lm_config.get('embedding_model_id')
            generation_model_id = lm_config.get('generation_model_id')
            api_base = lm_config.get('api_base')
            embedding_dim = lm_config.get('embedding_dim')
            if embedding_model_id:
                self.cortex_configs['language_cortex'] = {"type": "LanguageCortex", "params": {"model_id": embedding_model_id, "api_base": api_base, "embedding_dim": embedding_dim}}

        # --- Initialize Architecture Components ---
        self.cortexes = _initialize_cortexes(self.cortex_configs, self.max_obs_dim)

        # World Model (RSSM-based)
        self.world_model = WorldModel(self.max_obs_dim, self.max_action_dim, latent_dim, hidden_dim, hyperparams=self.hyperparams)

        # STAG Framework
        # The STAG still operates on the deterministic hidden state `h_t`
        self.stag = STAG_Framework(self.hidden_dim, **self.hyperparams)

        # STAG Context Processor
        # Input dimension is the max path length * the dimension of a node's weight vector (hidden_dim)
        self.stag_context_processor = StagContextProcessor(
            input_dim=self.max_stag_path_length * self.hidden_dim,
            output_dim=self.stag_context_dim
        )

        # Actor-Critic Planner
        # Actor (Policy Head)
        self.action_head = ActionHead(
            input_dim=self.hidden_dim + self.stag_context_dim,  # Takes h_t and C_t
            n_actions=self.max_action_dim,
            learning_rate=self.learning_rate
        )
        # Critic (Value Head)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, 400), # Takes h_t and z_t
            nn.ReLU(),
            nn.Linear(400, 1)
        )

        self.text_generation_head = None
        if self.language_model_enabled and generation_model_id:
            self.text_generation_head = TextGenerationHead(
                model_path_or_id=generation_model_id,
                input_dim=self.hidden_dim,
                api_base=api_base
            )

        if load_from_storage and self.history_manager._read_history():
            self.load_state()
        else:
            self.save_state(version_info={"message": "Initial state."})

        buffer_capacity = self.hyperparams.get('buffer_capacity', 10000)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def save_state(self, version_info={}):
        """Saves the agent's core models to a new version snapshot."""
        learnable_params = {
            'world_model_state_dict': self.world_model.state_dict(),
            'action_head_state_dict': self.action_head.state_dict(),
            'value_head_state_dict': self.value_head.state_dict(),
            'stag_context_processor_state_dict': self.stag_context_processor.state_dict(),
        }
        self.history_manager.save_snapshot(learnable_params, version_info)

    def load_state(self, version='latest'):
        """Loads the agent's core models from a version snapshot."""
        learnable_params = self.history_manager.load_snapshot(version)
        if learnable_params:
            self.world_model.load_state_dict(learnable_params['world_model_state_dict'])
            self.action_head.load_state_dict(learnable_params['action_head_state_dict'])
            self.value_head.load_state_dict(learnable_params['value_head_state_dict'])
            self.stag_context_processor.load_state_dict(learnable_params.get('stag_context_processor_state_dict')) # Use .get for backward compatibility
        else:
            print("Warning: No state to load.")

    def perceive_and_update_state(self, cortex_id, raw_obs, damage_taken=0):
        """
        Processes observation, updates vitals, and updates the world model state.
        """
        # 1. Update vitals based on damage from the last step
        self._update_vitals(damage_taken=damage_taken)

        # 2. Process raw observation through the specified cortex
        obs_numpy = self.cortexes[cortex_id].process(raw_obs)
        obs_tensor = torch.from_numpy(obs_numpy).float().unsqueeze(0)

        # 3. Use the world model to update the agent's internal state (h_t, z_t)
        with torch.no_grad():
            # The world model's forward pass is now just for training. We need to call the rssm directly.
            h_next, z_next, _ = self.world_model.rssm(obs_tensor, self.last_action, self.hidden_state, self.latent_state)

        # 4. Update the agent's internal state
        self.hidden_state = h_next
        self.latent_state = z_next

        # 5. The STAG framework organizes the agent's deterministic hidden state
        h_numpy = self.hidden_state.detach().numpy().flatten()
        norm = np.linalg.norm(h_numpy)
        h_normalized = h_numpy / norm if norm > 0 else h_numpy

        # Find the activation path and get the potential novelty signal (error)
        terminal_node, winner_id, activation_path = self.stag.find_terminal_node_and_path(h_normalized)
        novelty = 0
        if terminal_node and winner_id is not None:
            novelty = terminal_node['gng'].nodes[winner_id].get('error', 0)

        # The GNG is now updated in a separate step after the reward is known.
        return self.hidden_state, self.latent_state, h_normalized, activation_path, novelty

    def update_stag(self, h_normalized, r_env):
        """
        Updates the STAG/GNG with the given state and the environmental reward it produced.
        This is called in the main training loop after a reward is received.
        """
        terminal_node, _, _ = self.stag.find_terminal_node_and_path(h_normalized)
        if terminal_node:
            # The GNG's utility update is driven by the environmental reward
            terminal_node['gng'].process_input(h_normalized, reward=r_env)

    def select_action(self, actual_action_dim, activation_path):
        """
        Selects an action using the policy π(a | h_t, C_t), with masking for the action space.
        """
        # 1. Construct the context vector C_t from the STAG's activation path
        path_weights = []
        if activation_path:
            for step in activation_path:
                level_gng = self.stag.level_map[step['level_id']]
                node_weight = level_gng.nodes[step['winner_id']]['weight']
                path_weights.append(torch.from_numpy(node_weight).float())

        # Pad the path to a fixed length
        while len(path_weights) < self.max_stag_path_length:
            path_weights.append(torch.zeros(self.hidden_dim))

        # Process the path to get the context vector C_t
        with torch.no_grad():
            stag_context_vector = self.stag_context_processor(path_weights)

        # 2. The ActionHead receives the deterministic state h_t and context C_t
        combined_input = torch.cat((self.hidden_state, stag_context_vector), dim=1)

        # 3. Get action distribution, mask it, and select action
        with torch.no_grad():
            # Get the raw logits from the action head's linear layer
            logits = self.action_head.layer(combined_input)

            # Create a mask to disable logits for actions outside the valid range
            mask = torch.full(logits.shape, -float('inf'))
            mask[0, :actual_action_dim] = 0

            # Apply the mask to the logits
            masked_logits = logits + mask

            # Create the distribution from the masked logits
            action_dist = Categorical(logits=masked_logits)

        epsilon = self._get_epsilon()
        self.steps_done += 1

        if np.random.rand() < epsilon:
            action_tensor = torch.tensor([np.random.randint(0, actual_action_dim)])
        else:
            action_tensor = action_dist.sample()

        action = action_tensor.item()
        log_prob = action_dist.log_prob(action_tensor)

        self.last_action = action_tensor

        return action, log_prob, stag_context_vector

    def _get_epsilon(self):
        epsilon_start = self.hyperparams.get('epsilon_start', 0.9)
        epsilon_end = self.hyperparams.get('epsilon_end', 0.05)
        epsilon_decay_steps = self.hyperparams.get('epsilon_decay_steps', 20000)
        if self.steps_done < epsilon_decay_steps:
            return epsilon_start - (epsilon_start - epsilon_end) * (self.steps_done / epsilon_decay_steps)
        return epsilon_end

    def _update_vitals(self, damage_taken=0, energy_gain=0):
        """Updates agent's vitals."""
        # Update Integrity based on damage
        self.integrity = max(0, self.integrity - damage_taken)
        # Update Energy
        self.energy = min(self.max_energy, self.energy - self.metabolic_cost + energy_gain)

    def get_internal_reward(self, damage_taken, novelty_signal):
        """Calculates the total internal reward signal."""
        internal_reward = -self.metabolic_cost # Base metabolic cost

        # Penalty for low energy
        low_energy_threshold = self.hyperparams.get('low_energy_threshold', 0.2)
        if (self.energy / self.max_energy) < low_energy_threshold:
            internal_reward += self.hyperparams.get('low_energy_penalty', -10.0)

        # Penalty for taking damage
        damage_penalty_multiplier = self.hyperparams.get('damage_penalty_multiplier', -5.0)
        internal_reward += damage_penalty_multiplier * damage_taken

        # Reward for novelty (exploring new concepts in STAG)
        novelty_reward_weight = self.hyperparams.get('novelty_reward_weight', 0.1)
        internal_reward += novelty_reward_weight * novelty_signal

        return internal_reward

    def record_experience(self, *args):
        """Pushes an experience to the replay buffer."""
        # The experience tuple will need to be updated for the new training regime
        self.replay_buffer.push(*args)

    def train(self, cortex_id="vector_input"):
        """
        The main training loop that orchestrates the "sleep" phase of the agent,
        which involves training the world model and then training the policy in imagination.
        """
        # Part 1: Train the World Model on real, recently collected data.
        world_model_stats = self.train_world_model(cortex_id)

        # Part 2: Train the Actor-Critic policy in imagined trajectories.
        policy_stats = self.train_policy_in_imagination()

        # Combine stats for logging
        combined_stats = {**world_model_stats, **policy_stats}
        return combined_stats

    def train_world_model(self, cortex_id="vector_input"):
        """
        Trains the World Model (RSSM, Obs/Reward Decoders) on a batch of real data.
        This is the first part of the "Sleep" phase.
        """
        if cortex_id not in self.cortexes:
            logger.error(f"Invalid cortex_id '{cortex_id}' provided for training.")
            return {}

        cortex = self.cortexes[cortex_id]
        if not isinstance(cortex, torch.nn.Module):
            logger.warning(f"Cortex '{cortex_id}' is not trainable. Only training WorldModel.")
            # For non-trainable cortexes, we can still proceed
            pass

        batch_size = self.hyperparams.get('batch_size', 32)
        if len(self.replay_buffer) < batch_size:
            return {"status": "Not enough experiences in buffer to train."}

        # --- Sample a batch and prepare tensors ---
        experiences = self.replay_buffer.sample(batch_size)
        batch = Experience(*zip(*experiences))

        # The replay buffer stores raw numpy observations. Process them.
        def pad_observations(obs_list):
            padded_obs = [np.zeros(self.max_obs_dim) for _ in range(len(obs_list))]
            for i, o in enumerate(obs_list):
                padded_obs[i][:o.shape[0]] = o
            return torch.from_numpy(np.array(padded_obs)).float()

        # We need the *next* observation to calculate reconstruction loss
        next_obs_batch_raw = pad_observations(batch.next_obs)

        # Process observations through the cortex
        obs_batch_processed = cortex(next_obs_batch_raw) # Note: we are predicting the *next* observation

        h_prev_batch = torch.stack(batch.h).squeeze(1)
        z_prev_batch = torch.stack(batch.z).squeeze(1)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward).float()

        # --- Optimizers ---
        wm_params = list(self.world_model.parameters())
        if isinstance(cortex, torch.nn.Module):
            wm_params += list(cortex.parameters())
        wm_optimizer = torch.optim.Adam(wm_params, lr=self.hyperparams.get('world_model_lr', 0.001))

        # --- Forward Pass ---
        obs_recon, reward_pred, kl_loss, _, _ = self.world_model(
            obs_batch_processed, action_batch, h_prev_batch, z_prev_batch
        )

        # --- Loss Calculation (ℒ_WM) ---
        recon_loss = torch.nn.functional.mse_loss(obs_recon, obs_batch_processed)
        reward_loss = torch.nn.functional.mse_loss(reward_pred, reward_batch.unsqueeze(1))
        
        w_recon = self.hyperparams.get('w_recon', 1.0)
        w_reward = self.hyperparams.get('w_reward', 1.0)
        w_kl = self.hyperparams.get('w_kl', 1.0)

        world_model_loss = w_recon * recon_loss + w_reward * reward_loss + w_kl * kl_loss

        # --- Backpropagation ---
        wm_optimizer.zero_grad()
        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(wm_params, self.max_grad_norm)
        wm_optimizer.step()

        return {
            "wm_loss": world_model_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "kl_loss": kl_loss.item()
        }

    def train_policy_in_imagination(self):
        """
        Trains the Actor (ActionHead) and Critic (ValueHead) in imagination.
        This is the second part of the "Sleep" phase.
        """
        batch_size = self.hyperparams.get('batch_size', 32)
        if len(self.replay_buffer) < batch_size:
            return {}

        horizon = self.hyperparams.get('imagine_horizon', 15)

        # --- Sample starting states from real data ---
        experiences = self.replay_buffer.sample(batch_size)
        batch = Experience(*zip(*experiences))
        h_start = torch.stack(batch.h).squeeze(1)
        z_start = torch.stack(batch.z).squeeze(1)

        # --- Imagine Trajectories ---
        h_t, z_t = h_start, z_start
        imagined_h = [h_t]
        imagined_z = [z_t]
        imagined_actions = []
        imagined_stag_contexts = []

        # --- Imagine Trajectories with Dynamic STAG Context ---
        with torch.no_grad():
            for _ in range(horizon):
                # 1. Generate STAG context dynamically for the current imagined state h_t
                stag_contexts = []
                for i in range(h_t.size(0)): # Process each state in the batch
                    h_numpy = h_t[i].detach().numpy().flatten()
                    norm = np.linalg.norm(h_numpy)
                    h_normalized = h_numpy / norm if norm > 0 else h_numpy

                    _, _, activation_path = self.stag.find_terminal_node_and_path(h_normalized)

                    path_weights = []
                    if activation_path:
                        for step in activation_path:
                            level_gng = self.stag.level_map[step['level_id']]
                            node_weight = level_gng.nodes[step['winner_id']]['weight']
                            path_weights.append(torch.from_numpy(node_weight).float())

                    while len(path_weights) < self.max_stag_path_length:
                        path_weights.append(torch.zeros(self.hidden_dim))

                    # Note: The context processor expects a batch, but we process one by one
                    # This is less efficient but conceptually simpler for now.
                    stag_contexts.append(self.stag_context_processor(path_weights))

                stag_context_batch = torch.cat(stag_contexts, dim=0)


                # 2. Actor selects action based on h_t and the dynamically generated C_t
                action_input = torch.cat([h_t, stag_context_batch], dim=1)
                action_dist = self.action_head(action_input)
                action = action_dist.sample()

                # 3. World model predicts next state
                h_t, prior_mean, prior_std = self.world_model.rssm.transition_model(z_t, action, h_t)
                z_t = Normal(prior_mean, prior_std).rsample()

                imagined_h.append(h_t)
                imagined_z.append(z_t)
                imagined_actions.append(action)
                imagined_stag_contexts.append(stag_context_batch)

        imagined_h = torch.stack(imagined_h) # Shape: [horizon+1, batch, hidden_dim]
        imagined_z = torch.stack(imagined_z) # Shape: [horizon+1, batch, latent_dim]
        imagined_actions = torch.stack(imagined_actions) # Shape: [horizon, batch]
        imagined_stag_contexts = torch.stack(imagined_stag_contexts) # Shape: [horizon, batch, context_dim]


        # --- Predict Rewards and Values for Imagined Trajectory ---
        imagined_rewards = self.world_model.reward_model(imagined_z, imagined_h).squeeze(-1)
        imagined_values = self.value_head(torch.cat([imagined_h, imagined_z], dim=-1)).squeeze(-1)

        # --- Calculate Value Targets (Lambda-Return) ---
        lambda_ = self.hyperparams.get('lambda', 0.95)
        returns = torch.zeros_like(imagined_values[-1])
        lambda_returns = []
        for t in reversed(range(horizon)):
            # V_target = r_t + gamma * ( (1-lambda) * V(s_{t+1}) + lambda * V_target_{t+1} )
            returns = imagined_rewards[t] + self.gamma * ((1 - lambda_) * imagined_values[t+1] + lambda_ * returns)
            lambda_returns.append(returns)
        lambda_returns = torch.stack(list(reversed(lambda_returns)))

        # --- Actor-Critic Loss Calculation (ℒ_AC) ---
        # Actor Loss
        advantage = (lambda_returns - imagined_values[:-1]).detach()
        action_input = torch.cat([imagined_h[:-1], imagined_stag_contexts], dim=-1)
        log_probs = self.action_head.get_log_probs(action_input, imagined_actions)
        policy_loss = -(log_probs * advantage).mean()

        # Critic Loss
        critic_loss = torch.nn.functional.mse_loss(imagined_values[:-1], lambda_returns.detach())

        # Entropy Bonus
        entropy = self.action_head.get_entropy(action_input).mean()

        # --- Total AC Loss and Backpropagation ---
        w_policy = self.hyperparams.get('w_policy', 1.0)
        w_critic = self.hyperparams.get('w_critic', 0.5)
        w_entropy = self.hyperparams.get('w_entropy', 0.001)

        ac_loss = w_policy * policy_loss + w_critic * critic_loss - w_entropy * entropy

        ac_optimizer = torch.optim.Adam(
            list(self.action_head.parameters()) + list(self.value_head.parameters()),
            lr=self.hyperparams.get('actor_critic_lr', 0.0003)
        )
        ac_optimizer.zero_grad()
        ac_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.action_head.parameters()) + list(self.value_head.parameters()), self.max_grad_norm)
        ac_optimizer.step()

        return {
            "ac_loss": ac_loss.item(),
            "policy_loss": policy_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item()
        }

    def generate_response(self, max_new_tokens=50):
        """
        Generates a natural language response based on the agent's current state.
        """
        if not self.language_model_enabled or not self.text_generation_head:
            return "I am currently unable to speak."

        # Construct a dictionary of the agent's current state and vitals
        agent_state = {
            "vitals": {
                "energy": round(self.energy, 2),
                "integrity": round(self.integrity, 2),
            },
            "state_summary": {
                "mean": round(self.hidden_state.mean().item(), 4),
                "max": round(self.hidden_state.max().item(), 4),
                "min": round(self.hidden_state.min().item(), 4),
                "std": round(self.hidden_state.std().item(), 4),
            }
        }

        return self.text_generation_head.generate(
            agent_state,
            max_new_tokens=max_new_tokens
        )

    def get_graph_structure(self):
        return self.stag.get_flattened_structure()