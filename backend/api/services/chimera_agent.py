# Implements the ChimeraAgent, the core class for a single autonomous agent.
# This class encapsulates the agent's "brain", its sensory cortexes, and its
# action-selection mechanism.

import os
import json
import numpy as np
from .hopfield_core import HopfieldCore
from .stag_framework import STAG_Framework
from .cortex import modules as cortex_modules
from .action.modules import ActionHead

def _initialize_cortexes(configs, output_dim):
    """Helper function to instantiate cortex modules from a config dict."""
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
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class ChimeraAgent:
    def __init__(self, agent_id, dimensions=64, n_actions=256, cortex_configs=None, load_from_storage=True, **hyperparams):
        self.agent_id = agent_id
        self.dimensions = dimensions
        self.n_actions = n_actions
        self.storage_path = os.path.join('backend', 'storage', f'{self.agent_id}.npz')
        self.learning_rate = hyperparams.get('learning_rate', 0.01)
        self.gamma = hyperparams.get('gamma', 0.99) # Discount factor for rewards

        if load_from_storage and os.path.exists(self.storage_path):
            self.load_state()
        else:
            self.cortex_configs = cortex_configs or {}
            self.cortexes = _initialize_cortexes(self.cortex_configs, self.dimensions)
            self.hopfield = HopfieldCore(dimensions, **hyperparams)
            self.stag = STAG_Framework(dimensions, **hyperparams)
            self.action_head = ActionHead(input_dim=dimensions, n_actions=n_actions)

        self.episode_memory = [] # Stores (internal_state, action, reward)

    def save_state(self):
        stag_state_json = json.dumps(self.stag.get_serializable_structure())
        cortex_configs_json = json.dumps(self.cortex_configs)

        state_data = {
            'agent_id': self.agent_id, 'dimensions': self.dimensions, 'n_actions': self.n_actions,
            'cortex_configs_json': cortex_configs_json, 'hopfield_weights': self.hopfield.weights,
            'action_head_weights': self.action_head.weights, 'action_head_biases': self.action_head.biases,
            'stag_state_json': stag_state_json
        }
        np.savez_compressed(self.storage_path, **state_data)

    def load_state(self):
        with np.load(self.storage_path, allow_pickle=True) as data:
            self.dimensions = int(data['dimensions'])
            self.n_actions = int(data['n_actions'])
            cortex_configs_json = str(data['cortex_configs_json'])
            self.cortex_configs = json.loads(cortex_configs_json)
            self.cortexes = _initialize_cortexes(self.cortex_configs, self.dimensions)
            self.hopfield = HopfieldCore.from_state({'dimensions': self.dimensions, 'weights': data['hopfield_weights']})
            self.action_head = ActionHead(self.dimensions, self.n_actions)
            self.action_head.set_state({'weights': data['action_head_weights'], 'biases': data['action_head_biases']})
            stag_state_json = str(data['stag_state_json'])
            self.stag = STAG_Framework.from_serializable_structure(json.loads(stag_state_json))

    def perceive(self, cortex_id, raw_input):
        if cortex_id not in self.cortexes: raise ValueError(f"Cortex '{cortex_id}' not found.")
        return self.cortexes[cortex_id].process(raw_input)

    def get_internal_state_representation(self, input_embedding):
        stable_attractor = self.hopfield.recall(input_embedding)
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
        """
        Trains the agent using the experiences collected in an episode (REINFORCE).
        """
        # Calculate discounted returns
        discounted_rewards = []
        cumulative_reward = 0
        for experience in reversed(self.episode_memory):
            cumulative_reward = experience['reward'] + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        # Normalize rewards for stability
        rewards_mean = np.mean(discounted_rewards)
        rewards_std = np.std(discounted_rewards)
        normalized_rewards = (discounted_rewards - rewards_mean) / (rewards_std + 1e-7)

        # Update parameters
        for experience, G_t in zip(self.episode_memory, normalized_rewards):
            internal_state = experience['state']
            action = experience['action']

            # --- Update Action Head ---
            # This is the core policy gradient update rule
            action_logits = self.action_head.forward(internal_state)
            action_probs = softmax(action_logits)

            # Gradient of the softmax output for the chosen action
            d_softmax = action_probs
            d_softmax[action] -= 1

            # Gradient of the loss w.r.t. logits is G_t * d_softmax
            d_logits = G_t * d_softmax

            # Backpropagate to weights and biases
            d_weights = np.outer(internal_state, d_logits)
            d_biases = d_logits

            self.action_head.weights -= self.learning_rate * d_weights
            self.action_head.biases -= self.learning_rate * d_biases

            # --- Update GNG Node (Heuristic) ---
            # This is the simplified, non-backprop update for the brain itself.
            # We nudge the GNG node that was responsible for the action.
            # A positive G_t means the outcome was good, so we want the GNG node
            # to be more like the state that led to it.
            # This is a complex area; for now, we assume internal_state is the GNG node's weight vector
            # and we don't update it, as it's the target. A more advanced agent might
            # also update the Hopfield network based on reward. This part is left for future work.

        # Clear memory for the next episode
        self.episode_memory = []

        # Persist the updated agent state
        self.save_state()
        return {"status": "Training complete"}
