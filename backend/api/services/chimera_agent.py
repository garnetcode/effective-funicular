# Implements the ChimeraAgent, the core class for a single autonomous agent.
# This class encapsulates the agent's "brain", its sensory cortexes, and its
# action-selection mechanism.

import os
import json
import numpy as np

class NumpyJSONEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

from .hopfield_core import HopfieldCore
from .stag_framework import STAG_Framework
from .cortex import modules as cortex_modules
from .action.modules import ActionHead

try:
    import faiss
except ImportError:
    faiss = None

def _initialize_cortexes(configs, output_dim):
    # ... (same as before)
    cortexes = {}
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
    def __init__(self, agent_id, dimensions=64, n_actions=256, cortex_configs=None, load_from_storage=True, **hyperparams):
        self.agent_id = agent_id
        self.dimensions = dimensions
        self.n_actions = n_actions
        self.storage_path = os.path.join('backend', 'storage', f'{self.agent_id}.npz')
        self.faiss_index_path = os.path.join('backend', 'storage', f'{self.agent_id}.faiss')
        self.hyperparams = hyperparams
        self.learning_rate = self.hyperparams.get('learning_rate', 0.01)
        self.gamma = self.hyperparams.get('gamma', 0.99)

        if load_from_storage and os.path.exists(self.storage_path):
            self.load_state()
        else:
            self.cortex_configs = cortex_configs or {}
            self.cortexes = _initialize_cortexes(self.cortex_configs, self.dimensions)
            self.hopfield = HopfieldCore(dimensions, **self.hyperparams)
            self.stag = STAG_Framework(dimensions, **self.hyperparams)
            self.action_head = ActionHead(input_dim=dimensions, n_actions=n_actions)
            # A new agent's state must be saved immediately.
            self.save_state()

        self.episode_memory = []

    def save_state(self):
        # Save main agent data to .npz file
        stag_state_json = json.dumps(self.stag.get_serializable_structure(), cls=NumpyJSONEncoder)
        cortex_configs_json = json.dumps(self.cortex_configs, cls=NumpyJSONEncoder)
        hyperparams_json = json.dumps(self.hyperparams, cls=NumpyJSONEncoder)
        state_data = {
            'agent_id': self.agent_id, 'dimensions': self.dimensions, 'n_actions': self.n_actions,
            'cortex_configs_json': cortex_configs_json, 'hyperparams_json': hyperparams_json,
            'hopfield_weights': self.hopfield.weights,
            'action_head_weights': self.action_head.weights, 'action_head_biases': self.action_head.biases,
            'stag_state_json': stag_state_json
        }
        np.savez_compressed(self.storage_path, **state_data)

        # Save the FAISS index to a separate file
        if faiss and self.stag.tree['gng'].faiss_index:
            faiss.write_index(self.stag.tree['gng'].faiss_index, self.faiss_index_path)

    def load_state(self):
        # Load FAISS index first if it exists
        loaded_faiss_index = None
        if faiss and os.path.exists(self.faiss_index_path):
            loaded_faiss_index = faiss.read_index(self.faiss_index_path)

        # Load main agent data from .npz file
        with np.load(self.storage_path, allow_pickle=True) as data:
            self.dimensions = int(data['dimensions'])
            self.n_actions = int(data['n_actions'])

            hyperparams_json = str(data['hyperparams_json'])
            self.hyperparams = json.loads(hyperparams_json)
            self.learning_rate = self.hyperparams.get('learning_rate', 0.01)
            self.gamma = self.hyperparams.get('gamma', 0.99)

            cortex_configs_json = str(data['cortex_configs_json'])
            self.cortex_configs = json.loads(cortex_configs_json)
            self.cortexes = _initialize_cortexes(self.cortex_configs, self.dimensions)

            self.hopfield = HopfieldCore.from_state({
                'dimensions': self.dimensions, 'weights': data['hopfield_weights'], **self.hyperparams
            })
            self.action_head = ActionHead(self.dimensions, self.n_actions)
            self.action_head.set_state({'weights': data['action_head_weights'], 'biases': data['action_head_biases']})

            stag_state_json = str(data['stag_state_json'])
            stag_structure = json.loads(stag_state_json)
            # Pass the pre-loaded index to the constructor
            self.stag = STAG_Framework.from_serializable_structure(stag_structure, faiss_index=loaded_faiss_index, **self.hyperparams)

    # ... rest of the methods are the same ...
    def perceive(self, cortex_id, raw_input):
        if cortex_id not in self.cortexes: raise ValueError(f"Cortex '{cortex_id}' not found.")
        return self.cortexes[cortex_id].process(raw_input)

    def get_internal_state_representation(self, input_embedding):
        stable_attractor = self.hopfield.recall(input_embedding)
        # Assumes root GNG for now
        winner_id, _ = self.stag.tree['gng']._find_winners(stable_attractor)
        if winner_id is not None and winner_id in self.stag.tree['gng'].nodes:
            return self.stag.tree['gng'].nodes[winner_id]['weight']
        return stable_attractor

    def select_action(self, state_embedding):
        internal_state = self.get_internal_state_representation(state_embedding)
        action_logits = self.action_head.forward(internal_state)
        action_probs = softmax(action_logits)
        action = np.random.choice(self.n_actions, p=action_probs)
        log_prob = np.log(action_probs[action])
        return action, log_prob, internal_state

    def record_experience(self, internal_state, action, reward):
        self.episode_memory.append({"state": internal_state, "action": action, "reward": reward})

    def train(self):
        if not self.episode_memory: return {"status": "No experiences to train on."}

        discounted_rewards = []
        cumulative_reward = 0
        for experience in reversed(self.episode_memory):
            cumulative_reward = experience['reward'] + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        rewards_mean = np.mean(discounted_rewards)
        rewards_std = np.std(discounted_rewards)
        normalized_rewards = (discounted_rewards - rewards_mean) / (rewards_std + 1e-7)

        for experience, G_t in zip(self.episode_memory, normalized_rewards):
            internal_state = experience['state']
            action = experience['action']

            action_logits = self.action_head.forward(internal_state)
            action_probs = softmax(action_logits)
            d_softmax = action_probs
            d_softmax[action] -= 1
            d_logits = G_t * d_softmax

            d_weights = np.outer(internal_state, d_logits)
            d_biases = d_logits

            self.action_head.weights -= self.learning_rate * d_weights
            self.action_head.biases -= self.learning_rate * d_biases

        self.episode_memory = []
        self.save_state()
        return {"status": "Training complete"}

    def get_graph_structure(self):
        """
        Returns the hierarchical graph structure for frontend visualization.
        """
        return self.stag.get_serializable_structure()
