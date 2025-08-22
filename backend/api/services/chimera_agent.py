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

        self.storage_path = os.path.join('backend', 'storage', f'{self.agent_id}.npz')
        self.hyperparams = hyperparams
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

        if load_from_storage and os.path.exists(self.storage_path):
            self.load_state()
        else:
            self.cortex_configs = cortex_configs or {}
            # The world model's encoder handles the input, so cortexes now output to obs_dim
            self.cortexes = _initialize_cortexes(self.cortex_configs, self.obs_dim)

            self.world_model = WorldModel(obs_dim, action_dim, latent_dim, hidden_dim)

            # STAG now organizes the hidden states of the world model
            self.stag = STAG_Framework(self.hidden_dim, **self.hyperparams)

            self.action_head = ActionHead(
                input_dim=self.hidden_dim, # Action is based on the world model's hidden state
                n_actions=self.action_dim,
                learning_rate=self.learning_rate
            )

            # Data structures for STAG data partitioning
            self._next_pattern_id = 0
            self.patterns = {} # {pattern_id: hidden_state_vector}
            self.pattern_node_map = {}

            self.save_state()

        self.episode_memory = []

    def save_state(self):
        storage_dir = os.path.dirname(self.storage_path)
        os.makedirs(storage_dir, exist_ok=True)

        stag_state_json = json.dumps(self.stag.get_serializable_structure(), cls=NumpyJSONEncoder)
        cortex_configs_json = json.dumps(self.cortex_configs, cls=NumpyJSONEncoder)
        hyperparams_json = json.dumps(self.hyperparams, cls=NumpyJSONEncoder)
        patterns_json = json.dumps(self.patterns, cls=NumpyJSONEncoder)
        pattern_node_map_json = json.dumps(self.pattern_node_map, cls=NumpyJSONEncoder)

        state_data = {
            'agent_id': self.agent_id,
            'obs_dim': self.obs_dim, 'action_dim': self.action_dim,
            'latent_dim': self.latent_dim, 'hidden_dim': self.hidden_dim,
            'cortex_configs_json': cortex_configs_json, 'hyperparams_json': hyperparams_json,
            'world_model_state_dict': self.world_model.state_dict(),
            'action_head_state': self.action_head.get_state(),
            'stag_state_json': stag_state_json,
            'next_pattern_id': self._next_pattern_id,
            'patterns_json': patterns_json,
            'pattern_node_map_json': pattern_node_map_json,
            'energy': self.energy,
            'integrity': self.integrity,
        }
        np.savez_compressed(self.storage_path, **state_data)

    def load_state(self):
        with np.load(self.storage_path, allow_pickle=True) as data:
            self.obs_dim = int(data['obs_dim'])
            self.action_dim = int(data['action_dim'])
            self.latent_dim = int(data.get('latent_dim', 64)) # Backward compatibility
            self.hidden_dim = int(data.get('hidden_dim', 128)) # Backward compatibility

            self.hyperparams = json.loads(str(data['hyperparams_json']))
            self.learning_rate = self.hyperparams.get('learning_rate', 0.01)
            self.gamma = self.hyperparams.get('gamma', 0.99)
            self.stag_expansion_threshold = self.hyperparams.get('stag_expansion_threshold', 0.1)

            self.cortex_configs = json.loads(str(data['cortex_configs_json']))
            self.cortexes = _initialize_cortexes(self.cortex_configs, self.obs_dim)

            # Reconstruct World Model
            self.world_model = WorldModel(self.obs_dim, self.action_dim, self.latent_dim, self.hidden_dim)
            self.world_model.load_state_dict(data['world_model_state_dict'].item())

            self.action_head = ActionHead(
                self.hidden_dim,
                self.action_dim,
                learning_rate=self.learning_rate
            )
            self.action_head.set_state(data['action_head_state'].item())

            stag_structure = json.loads(str(data['stag_state_json']))
            self.stag = STAG_Framework.from_serializable_structure(stag_structure, **self.hyperparams)

            # Load pattern mapping data
            self._next_pattern_id = int(data.get('next_pattern_id', 0))
            self.patterns = json.loads(str(data.get('patterns_json', '{}')))
            self.patterns = {int(k): np.array(v) for k, v in self.patterns.items()} # Re-numpyfy
            self.pattern_node_map = json.loads(str(data.get('pattern_node_map_json', '{}')))
            # Handle backward compatibility for old pattern_node_map format
            temp_map = {}
            for k, v in self.pattern_node_map.items():
                key = int(k)
                if isinstance(v, int):
                    # Old format, assume it belongs to the root GNG (level 0)
                    temp_map[key] = (0, v)
                else:
                    # New format, already a tuple
                    temp_map[key] = tuple(v)
            self.pattern_node_map = temp_map

            # Load vitals with backward compatibility
            self.energy = float(data.get('energy', self.max_energy))
            self.integrity = float(data.get('integrity', 100.0))


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

        # --- Train the World Model ---
        world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.learning_rate)

        # Process a batch of experiences
        # Note: A more advanced implementation would use a replay buffer and batch sampling.
        # For simplicity, we train on the whole episode.
        obs_batch = torch.stack([torch.from_numpy(e['obs']) for e in self.episode_memory]).float()
        action_batch = torch.tensor([e['action'] for e in self.episode_memory])
        reward_batch = torch.tensor([e['reward'] for e in self.episode_memory]).float()
        next_obs_batch = torch.stack([torch.from_numpy(e['next_obs']) for e in self.episode_memory]).float()

        # We need to run the model step-by-step to get the hidden states
        # This is a simplified, non-batch version for clarity
        total_wm_loss = 0
        h = torch.zeros(1, self.hidden_dim) # Reset hidden state at start of sequence
        for i in range(len(self.episode_memory)):
            obs = obs_batch[i]
            action = action_batch[i]

            _, h, obs_pred, reward_pred = self.world_model(obs, action, h)

            # Calculate losses for this step
            reconstruction_loss = torch.nn.functional.mse_loss(obs_pred, next_obs_batch[i])
            reward_loss = torch.nn.functional.mse_loss(reward_pred, reward_batch[i].unsqueeze(0))
            total_wm_loss += reconstruction_loss + reward_loss

        world_model_optimizer.zero_grad()
        total_wm_loss.backward()
        world_model_optimizer.step()


        # --- Train the Action Head (Policy) ---
        # The policy is trained to act based on the world model's hidden states.
        # We need to re-calculate log_probs with the updated action head.
        policy_loss = 0
        h = torch.zeros(1, self.hidden_dim)
        for i in range(len(self.episode_memory)):
            obs = obs_batch[i]
            action = action_batch[i]
            reward = reward_batch[i] # In a full MBR-L setup, this could be the predicted reward

            # Get the hidden state for this observation
            with torch.no_grad():
                _, h, _, _ = self.world_model(obs, action, h)

            # Get the action distribution from this state
            action_logits = self.action_head(h)
            log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
            action_log_prob = log_probs.squeeze(0)[action]

            # Simple REINFORCE update for this step
            policy_loss -= action_log_prob * reward

        self.action_head.optimizer.zero_grad()
        policy_loss.backward()
        self.action_head.optimizer.step()


        # Clear memory for the next episode
        self.episode_memory = []
        self.save_state()
        return {
            "status": "Training complete",
            "world_model_loss": total_wm_loss.item(),
            "policy_loss": policy_loss.item()
        }

    def get_graph_structure(self):
        return self.stag.get_flattened_structure()
