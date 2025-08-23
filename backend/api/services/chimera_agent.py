# Implements the ChimeraAgent, the core orchestrator for the cognitive architecture.
# This class encapsulates the agent's "brain" (cognitive layers), its sensory
# cortexes, and its action-selection mechanism. It is responsible for managing
# the flow of information between the layers as specified in the Project Chimera doc.

import os
import json
import logging
import numpy as np
import torch

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
    def __init__(self, agent_id, max_obs_dim, max_action_dim, latent_dim=64, hidden_dim=128, cortex_configs=None, load_from_storage=True, hyperparams=None, history_config=None, **kwargs):
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
        self.learning_rate = self.hyperparams.get('learning_rate', 0.01)
        self.gamma = self.hyperparams.get('gamma', 0.99)
        self.stag_expansion_threshold = self.hyperparams.get('stag_expansion_threshold', 0.1)

        # --- Add Homeostatic Vitals ---
        self.max_energy = 100.0
        self.energy = self.max_energy
        self.integrity = 100.0
        self.metabolic_cost = 0.1

        # --- Initialize Agent State and Components ---
        self.hidden_state = torch.zeros(1, self.hidden_dim)
        self.last_action = torch.tensor(0)
        self.cortex_configs = cortex_configs or {}

        # Conditionally add language model config to cortex_configs
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
                self.cortex_configs['language_cortex'] = {
                    "type": "LanguageCortex",
                    "params": {
                        "model_id": embedding_model_id,
                        "api_base": api_base,
                        "embedding_dim": embedding_dim
                    }
                }

        # The agent's internal architecture is fixed to the max dimensions
        self.cortexes = _initialize_cortexes(self.cortex_configs, self.max_obs_dim)
        self.world_model = WorldModel(self.max_obs_dim, self.max_action_dim, latent_dim, hidden_dim)
        self.stag = STAG_Framework(self.hidden_dim, **self.hyperparams)
        self.action_head = ActionHead(
            input_dim=self.hidden_dim * 2,  # Hidden state + STAG context vector
            n_actions=self.max_action_dim,
            learning_rate=self.learning_rate
        )
        self.text_generation_head = None
        if self.language_model_enabled and generation_model_id:
            self.text_generation_head = TextGenerationHead(
                model_path_or_id=generation_model_id,
                input_dim=self.hidden_dim,
                api_base=api_base  # Pass Ollama API base
            )

        # Attempt to load saved state, otherwise save the initial state
        if load_from_storage and self.history_manager._read_history():
            self.load_state()
        else:
            self.save_state(version_info={"message": "Initial state."})

        # Initialize Replay Buffer for online learning
        buffer_capacity = self.hyperparams.get('buffer_capacity', 10000)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

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

    def select_action(self, actual_action_dim):
        """
        Selects an action based on the current internal state of the agent,
        augmented with context from the STAG knowledge graph, and constrained
        by the actual action space of the environment.
        """
        internal_state = self.hidden_state
        h_numpy = internal_state.detach().numpy().flatten()

        # Find the current conceptual context from the STAG
        terminal_level_node, winner_id, _ = self.stag.find_terminal_node_and_path(h_numpy)

        if winner_id is not None:
            stag_context_vector = terminal_level_node['gng'].nodes[winner_id]['weight']
            stag_context_vector = torch.from_numpy(stag_context_vector).float().unsqueeze(0)
        else:
            # If no specific node, use a zero vector as context
            stag_context_vector = torch.zeros(1, self.hidden_dim)

        # The ActionHead now receives both the transient hidden state and the stable STAG context
        combined_input = torch.cat((internal_state, stag_context_vector), dim=1)

        with torch.no_grad():
            action_logits = self.action_head(combined_input).squeeze(0)

        # Mask the logits to only consider valid actions for the current environment
        valid_logits = action_logits[:actual_action_dim]

        # Convert to numpy for softmax and sampling
        action_probs_np = softmax(valid_logits.numpy())
        action = np.random.choice(actual_action_dim, p=action_probs_np)
        action_tensor = torch.tensor(action)

        # We need the log_prob as a tensor for the loss calculation
        log_probs = torch.nn.functional.log_softmax(valid_logits, dim=-1)
        action_log_prob = log_probs[action]

        # Store the chosen action for the next world model update
        self.last_action = action_tensor

        return action, action_log_prob, stag_context_vector

    def record_experience(self, *args):
        """Pushes an experience to the replay buffer."""
        self.replay_buffer.push(*args)

    def train(self, cortex_id="vector_input"):
        """
        Samples a batch from the replay buffer and performs one step of training
        for the World Model, the specified Cortex, and the Action Head.
        """
        if cortex_id not in self.cortexes:
            logger.error(f"Invalid cortex_id '{cortex_id}' provided for training.")
            return {"status": f"Invalid cortex_id '{cortex_id}'"}

        cortex = self.cortexes[cortex_id]
        if not isinstance(cortex, torch.nn.Module):
            # This cortex is not trainable, so we can't proceed with joint training.
            # We will train the world model on the pre-processed observations.
            logger.warning(f"Cortex '{cortex_id}' is not a trainable torch.nn.Module. Only training WorldModel and ActionHead.")
            # TODO: A separate training path could be implemented here if needed.
            return {"status": f"Cortex '{cortex_id}' is not trainable."}

        batch_size = self.hyperparams.get('batch_size', 32)
        if len(self.replay_buffer) < batch_size:
            return {"status": "Not enough experiences in buffer to train."}

        # --- Sample a batch and prepare tensors ---
        experiences = self.replay_buffer.sample(batch_size)
        batch = Experience(*zip(*experiences))

        hidden_state_batch = torch.stack([h.detach() for h in batch.hidden_state]).squeeze(1)
        stag_context_batch = torch.stack([s.detach() for s in batch.stag_context]).squeeze(1)

        # The replay buffer stores raw numpy observations.
        # We need to pad them and convert to a tensor for the cortex.
        def pad_observations(obs_list):
            padded_obs = []
            for o in obs_list:
                padded = np.zeros(self.max_obs_dim)
                padded[:o.shape[0]] = o
                padded_obs.append(padded)
            return torch.from_numpy(np.array(padded_obs)).float()

        obs_batch_raw = pad_observations(batch.obs)
        next_obs_batch_raw = pad_observations(batch.next_obs)

        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward).float()

        # --- Initialize Optimizers ---
        # Combine parameters of the world model and the cortex for joint training
        world_model_and_cortex_params = list(self.world_model.parameters()) + list(cortex.parameters())
        world_model_optimizer = torch.optim.Adam(world_model_and_cortex_params, lr=self.learning_rate)
        action_head_optimizer = self.action_head.optimizer

        # --- World Model and Cortex Training ---
        # Process raw observations through the cortex first
        obs_batch_processed = cortex(obs_batch_raw)
        with torch.no_grad(): # Don't need gradients for the target
            next_obs_batch_processed = cortex(next_obs_batch_raw)

        # The world model predicts the next processed state from the processed current state
        _, h_next, obs_pred_batch, reward_pred_batch = self.world_model(obs_batch_processed, action_batch, hidden_state_batch)

        reconstruction_loss = torch.nn.functional.mse_loss(obs_pred_batch, next_obs_batch_processed)
        reward_loss = torch.nn.functional.mse_loss(reward_pred_batch, reward_batch.unsqueeze(1))
        total_wm_loss = reconstruction_loss + reward_loss

        world_model_optimizer.zero_grad()
        total_wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(world_model_and_cortex_params, max_norm=1.0)
        world_model_optimizer.step()

        # --- Action Head Training ---
        # The ActionHead is trained on the combined state (WorldModel hidden state + STAG context)
        combined_input_batch = torch.cat((hidden_state_batch, stag_context_batch), dim=1)
        action_logits = self.action_head(combined_input_batch)
        log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
        log_probs_for_actions = log_probs.gather(1, action_batch.unsqueeze(1))

        # The reward is the discounted return G_t calculated before buffering
        policy_loss = (-log_probs_for_actions * reward_batch.unsqueeze(1)).mean()

        action_head_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.action_head.parameters(), max_norm=1.0)
        action_head_optimizer.step()

        return {
            "status": "Training step complete",
            "world_model_loss": total_wm_loss.item(),
            "policy_loss": policy_loss.item()
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
