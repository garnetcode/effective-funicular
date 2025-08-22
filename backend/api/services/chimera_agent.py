# Implements the ChimeraAgent, the core orchestrator for the cognitive architecture.
# This class encapsulates the agent's "brain" (cognitive layers), its sensory
# cortexes, and its action-selection mechanism. It is responsible for managing
# the flow of information between the layers as specified in the Project Chimera doc.

import os
import json
import numpy as np
import torch

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
from .cortex import modules as cortex_modules
from .action.modules import ActionHead

def _initialize_cortexes(configs, output_dim):
    cortexes = {}
    if configs is None: return cortexes
    for cortex_id, config in configs.items():
        class_name = config['type']
        params = config.get('params', {})
        try:
            CortexClass = getattr(cortex_modules, class_name)
            if class_name == "DenseCortex":
                cortexes[cortex_id] = CortexClass(input_dim=params['input_dim'], output_dim=output_dim)
            else:
                cortexes[cortex_id] = CortexClass(output_dim=output_dim)
        except (AttributeError, ImportError) as e:
            print(f"Warning: Could not initialize cortex '{cortex_id}' of type '{class_name}': {e}")
    return cortexes

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class ChimeraAgent:
    def __init__(self, agent_id, obs_dim, action_dim, latent_dim=64, hidden_dim=128, cortex_configs=None, load_from_storage=True, **hyperparams):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.hyperparams = hyperparams
        self.history_manager = StateHistoryManager(agent_id)
        self.learning_rate = self.hyperparams.get('learning_rate', 0.01)
        self.gamma = self.hyperparams.get('gamma', 0.99)
        self.stag_expansion_threshold = self.hyperparams.get('stag_expansion_threshold', 0.1)

        # --- Add Homeostatic Vitals ---
        self.max_energy = 100.0
        self.energy = self.max_energy
        self.integrity = 100.0
        # Define the metabolic cost for just existing for one step
        self.metabolic_cost = 0.1 # This is a small, constant energy drain

        # The agent's current hidden state for the world model
        self.hidden_state = torch.zeros(1, self.hidden_dim)
        # Last action taken, for world model input
        self.last_action = torch.tensor(0)

        # Attempt to load the latest state if it exists
        if load_from_storage and self.history_manager._read_history():
            self.load_state()
        else:
            # Initialize a new agent state
            self.cortex_configs = cortex_configs or {}
            self.cortexes = _initialize_cortexes(self.cortex_configs, self.obs_dim)
            self.world_model = WorldModel(obs_dim, action_dim, latent_dim, hidden_dim)
            self.stag = STAG_Framework(self.hidden_dim, **self.hyperparams)
            self.action_head = ActionHead(
                input_dim=self.hidden_dim,
                n_actions=self.action_dim,
                learning_rate=self.learning_rate
            )
            # Save the initial state as version 0
            self.save_state(version_info={"message": "Initial state."})

        self.episode_memory = []

    def save_state(self, version_info={}):
        """Saves the agent's core models to a new version snapshot."""
        # Consolidate all learnable model parameters into one dictionary for versioning
        learnable_params = {
            'world_model_state_dict': self.world_model.state_dict(),
            'action_head_state_dict': self.action_head.state_dict(), # Use state_dict for weights only
            # TODO: Add STAG and other components to versioning if they become learnable
        }
        self.history_manager.save_snapshot(learnable_params, version_info)

    def load_state(self, version='latest'):
        """Loads the agent's core models from a version snapshot."""
        learnable_params = self.history_manager.load_snapshot(version)

        if learnable_params:
            self.world_model.load_state_dict(learnable_params['world_model_state_dict'])
            self.action_head.load_state_dict(learnable_params['action_head_state_dict'])
            # TODO: Load STAG state etc.
        else:
            # This should only happen if there's no history
            print("Warning: No state to load.")


    def perceive_and_update_state(self, cortex_id, raw_obs):
        """
        Processes a raw observation through a cortex, then updates the world
        model's hidden state.
        """
        # 1. Process raw observation through the specified cortex
        obs_numpy = self.cortexes[cortex_id].process(raw_obs)
        obs_tensor = torch.from_numpy(obs_numpy).float()

        # 2. Use the world model to update the hidden state
        # The 'action' is the last action taken to get to this new observation.
        with torch.no_grad():
            _, h_next, _, _ = self.world_model(obs_tensor, self.last_action, self.hidden_state)

        # 3. Update the agent's internal state
        self.hidden_state = h_next

        # The STAG framework now organizes the agent's internal, context-rich hidden states
        h_numpy = self.hidden_state.detach().numpy().flatten()

        # Find the correct terminal GNG and process the input
        terminal_node, _, _ = self.stag.find_terminal_node_and_path(h_numpy)
        if terminal_node:
            terminal_node['gng'].process_input(h_numpy)

        return self.hidden_state

    def select_action(self):
        """
        Selects an action based on the current internal state of the agent.
        """
        # The internal state for action selection is the world model's hidden state
        internal_state = self.hidden_state

        with torch.no_grad():
            action_logits = self.action_head(internal_state).squeeze(0)

        # Convert to numpy for softmax and sampling
        action_probs_np = softmax(action_logits.numpy())
        action = np.random.choice(self.action_dim, p=action_probs_np)
        action_tensor = torch.tensor(action)

        # We need the log_prob as a tensor for the loss calculation
        log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
        action_log_prob = log_probs[action]

        # Store the chosen action for the next world model update
        self.last_action = action_tensor

        return action, action_log_prob

    def record_experience(self, obs, action, log_prob, reward, next_obs, done):
        self.episode_memory.append({
            "obs": obs,
            "action": action,
            "log_prob": log_prob,
            "reward": reward,
            "next_obs": next_obs,
            "done": done
        })

    def train(self):
        if not self.episode_memory:
            return {"status": "No experiences to train on."}

        # --- Prepare Batches ---
        obs_batch = torch.stack([torch.from_numpy(e['obs']) for e in self.episode_memory]).float()
        action_batch = torch.tensor([e['action'] for e in self.episode_memory])
        reward_batch = torch.tensor([e['reward'] for e in self.episode_memory]).float()
        next_obs_batch = torch.stack([torch.from_numpy(e['next_obs']) for e in self.episode_memory]).float()

        # --- Initialize ---
        world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.learning_rate)
        action_head_optimizer = self.action_head.optimizer
        total_wm_loss = 0
        total_policy_loss = 0
        h = torch.zeros(1, self.hidden_dim) # Initial hidden state

        # --- Single Loop Training ---
        # We iterate through the episode once, calculating both losses simultaneously.
        for i in range(len(self.episode_memory)):
            # Detach hidden state to prevent gradients from flowing endlessly through time
            h = h.detach()

            # --- World Model Forward Pass and Loss ---
            _, h, obs_pred, reward_pred = self.world_model(obs_batch[i], action_batch[i], h)

            reconstruction_loss = torch.nn.functional.mse_loss(obs_pred, next_obs_batch[i])
            reward_loss = torch.nn.functional.mse_loss(reward_pred, reward_batch[i].unsqueeze(0))
            total_wm_loss += reconstruction_loss + reward_loss

            # --- Action Head Forward Pass and Loss ---
            # The policy acts on the hidden state that *produced* the action.
            # So we use the hidden state `h` we just calculated.
            action_logits = self.action_head(h)
            log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
            action_log_prob = log_probs.squeeze(0)[action_batch[i]]

            # Simple REINFORCE update (policy gradient)
            total_policy_loss -= action_log_prob * reward_batch[i]

        # --- Calculate Gradients ---
        world_model_optimizer.zero_grad()
        action_head_optimizer.zero_grad()

        # We calculate the gradients for both losses before updating any weights.
        # The backward passes accumulate gradients in the `.grad` attributes of the tensors.
        total_wm_loss.backward(retain_graph=True)
        mean_policy_loss = total_policy_loss / len(self.episode_memory)
        mean_policy_loss.backward()

        # --- Apply Gradients (Update Weights) ---
        world_model_optimizer.step()
        action_head_optimizer.step()

        # --- Cleanup ---
        self.episode_memory = []
        self.save_state()

        return {
            "status": "Training complete",
            "world_model_loss": total_wm_loss.item(),
            "policy_loss": mean_policy_loss.item()
        }

    def get_graph_structure(self):
        return self.stag.get_flattened_structure()
