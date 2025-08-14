# Implements the ChimeraAgent, the core orchestrator for the cognitive architecture.
# This class encapsulates the agent's "brain" (cognitive layers), its sensory
# cortexes, and its action-selection mechanism. It is responsible for managing
# the flow of information between the layers as specified in the Project Chimera doc.

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
        elif isinstance(obj, (np.float64, np.float16, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

from .hopfield_core import HopfieldCore
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
    def __init__(self, agent_id, dimensions=64, n_actions=256, cortex_configs=None, load_from_storage=True, **hyperparams):
        self.agent_id = agent_id
        self.dimensions = dimensions
        self.n_actions = n_actions
        self.storage_path = os.path.join('backend', 'storage', f'{self.agent_id}.npz')
        self.hyperparams = hyperparams
        self.learning_rate = self.hyperparams.get('learning_rate', 0.01)
        self.gamma = self.hyperparams.get('gamma', 0.99)
        self.stag_expansion_threshold = self.hyperparams.get('stag_expansion_threshold', 0.1)

        if load_from_storage and os.path.exists(self.storage_path):
            self.load_state()
        else:
            self.cortex_configs = cortex_configs or {}
            self.cortexes = _initialize_cortexes(self.cortex_configs, self.dimensions)
            self.hopfield = HopfieldCore(
                dimensions,
                learning_rate=self.hyperparams.get('learning_rate', 0.1),
                weight_decay=self.hyperparams.get('weight_decay', 0.01)
            )
            self.stag = STAG_Framework(dimensions, **self.hyperparams)
            self.action_head = ActionHead(input_dim=dimensions, n_actions=n_actions)

            # Data structures for STAG data partitioning
            self._next_pattern_id = 0
            self.patterns = {} # {pattern_id: pattern_vector}
            # {pattern_id: (tree_node_path, gng_node_id)} - simplified for now
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
            'agent_id': self.agent_id, 'dimensions': self.dimensions, 'n_actions': self.n_actions,
            'cortex_configs_json': cortex_configs_json, 'hyperparams_json': hyperparams_json,
            'hopfield_weights': self.hopfield.weights,
            'action_head_weights': self.action_head.weights, 'action_head_biases': self.action_head.biases,
            'stag_state_json': stag_state_json,
            'next_pattern_id': self._next_pattern_id,
            'patterns_json': patterns_json,
            'pattern_node_map_json': pattern_node_map_json,
        }
        np.savez_compressed(self.storage_path, **state_data)

    def load_state(self):
        with np.load(self.storage_path, allow_pickle=True) as data:
            self.dimensions = int(data['dimensions'])
            self.n_actions = int(data['n_actions'])
            self.hyperparams = json.loads(str(data['hyperparams_json']))
            self.learning_rate = self.hyperparams.get('learning_rate', 0.01)
            self.gamma = self.hyperparams.get('gamma', 0.99)
            self.stag_expansion_threshold = self.hyperparams.get('stag_expansion_threshold', 0.1)

            self.cortex_configs = json.loads(str(data['cortex_configs_json']))
            self.cortexes = _initialize_cortexes(self.cortex_configs, self.dimensions)

            # Reconstruct Hopfield Core by mapping keys correctly.
            # The hyperparams dictionary contains keys for all services, so we
            # must pass only the ones relevant to the HopfieldCore.
            hopfield_state = {
                'dimensions': self.dimensions,
                'weights': data['hopfield_weights'],
                'learning_rate': self.hyperparams.get('learning_rate', 0.1),
                'weight_decay': self.hyperparams.get('weight_decay', 0.01)
            }
            self.hopfield = HopfieldCore.from_state(hopfield_state)
            self.action_head = ActionHead(self.dimensions, self.n_actions)
            self.action_head.set_state({'weights': data['action_head_weights'], 'biases': data['action_head_biases']})

            stag_structure = json.loads(str(data['stag_state_json']))
            self.stag = STAG_Framework.from_serializable_structure(stag_structure, **self.hyperparams)

            # Load pattern mapping data
            self._next_pattern_id = int(data.get('next_pattern_id', 0))
            self.patterns = json.loads(str(data.get('patterns_json', '{}')))
            self.patterns = {int(k): np.array(v) for k, v in self.patterns.items()} # Re-numpyfy
            self.pattern_node_map = json.loads(str(data.get('pattern_node_map_json', '{}')))
            self.pattern_node_map = {int(k): v for k, v in self.pattern_node_map.items()} # Keys to int


    def learn_associative(self, embedding):
        """
        LEARN Mode (Sec 3.3): Learns a new pattern in the Hopfield core.
        """
        if not isinstance(embedding, np.ndarray): embedding = np.array(embedding)
        self.hopfield.learn(embedding)

        pattern_id = self._next_pattern_id
        self.patterns[pattern_id] = embedding
        self._next_pattern_id += 1

        self.save_state()
        return {"status": "Associative learning complete.", "pattern_id": pattern_id}

    def organize_memory(self, pattern_id):
        """
        ORGANIZE Mode (Sec 3.3): Updates the GNG/STAG structure.
        """
        if pattern_id not in self.patterns:
            raise ValueError(f"Pattern ID {pattern_id} not found.")

        cue_vector = self.patterns[pattern_id]

        # 1. Get stable attractor from Hopfield network
        stable_attractor = self.hopfield.recall(cue_vector)

        # 2. Find the terminal GNG and winner node for this attractor
        terminal_node, winner_id = self.stag.find_terminal_node(stable_attractor)
        if winner_id is None: # GNG not initialized enough
            terminal_node['gng'].process_input(stable_attractor)
            self.save_state()
            return {"status": "Organization step skipped, GNG too small."}

        # 3. Update the mapping for this pattern
        # Note: A real system would need a more robust path identifier
        self.pattern_node_map[pattern_id] = winner_id

        # 4. Process the input in the terminal GNG
        terminal_gng = terminal_node['gng']
        terminal_gng.process_input(stable_attractor)

        # 5. Check for expansion condition
        if terminal_gng.nodes[winner_id]['error'] > self.stag_expansion_threshold:
            self._trigger_expansion(terminal_node, winner_id)

        self.save_state()
        return {"status": "Organization step complete."}

    def _trigger_expansion(self, parent_level_node, parent_gng_node_id):
        """
        Orchestrates the expansion of a GNG node into a new child GNG.
        """
        # 1. Find all patterns that belong to the node being expanded
        patterns_for_new_gng = []
        # This is inefficient, a reverse map would be better in a real system
        for pid, mapped_nid in self.pattern_node_map.items():
            if mapped_nid == parent_gng_node_id:
                patterns_for_new_gng.append(self.patterns[pid])

        if not patterns_for_new_gng:
            print(f"Node {parent_gng_node_id} triggered expansion but no patterns mapped to it. Skipping.")
            return

        # 2. Create the new child GNG in the STAG framework
        new_child_gng = self.stag.expand_node(parent_level_node, parent_gng_node_id)
        if new_child_gng is None: return

        # 3. Train the new child GNG with the identified patterns (data partitioning)
        print(f"Training new child GNG with {len(patterns_for_new_gng)} patterns.")
        for pattern_vector in patterns_for_new_gng:
            # We use the original pattern vector's attractor state for training
            stable_attractor = self.hopfield.recall(pattern_vector)
            new_child_gng.process_input(stable_attractor)

            # Re-map the pattern to its new home in the child GNG
            new_winner_id, _ = new_child_gng._find_winners(stable_attractor)
            # This mapping needs to be more sophisticated to include hierarchy path
            # For now, we just overwrite it, which is incorrect for deep trees.
            # pattern_id = ... how to get pattern id here?
            # self.pattern_node_map[pattern_id] = new_winner_id
            # This part highlights need for further refinement in a real system.

    def update_cortex_config(self, new_cortex_configs):
        """Merges new cortex configs, re-initializes cortexes, and saves the agent."""
        self.cortex_configs.update(new_cortex_configs)
        self.cortexes = _initialize_cortexes(self.cortex_configs, self.dimensions)
        self.save_state()
        print(f"Agent {self.agent_id} cortex config updated to: {self.cortex_configs}")

    def update_action_space(self, n_actions):
        """Resets the agent's action head for a new number of actions."""
        self.n_actions = n_actions
        self.action_head = ActionHead(input_dim=self.dimensions, n_actions=self.n_actions)
        self.save_state()
        print(f"Agent {self.agent_id} action space updated to: {n_actions} actions")

    def perceive(self, cortex_id, raw_input):
        if cortex_id not in self.cortexes: raise ValueError(f"Cortex '{cortex_id}' not found.")
        return self.cortexes[cortex_id].process(raw_input)

    def get_internal_state_representation(self, input_embedding):
        stable_attractor = self.hopfield.recall(input_embedding)
        terminal_node, winner_id = self.stag.find_terminal_node(stable_attractor)
        if winner_id is not None and winner_id in terminal_node['gng'].nodes:
            return terminal_node['gng'].nodes[winner_id]['weight']
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
        # This is policy/RL training, separate from unsupervised cognitive learning
        if not self.episode_memory: return {"status": "No experiences to train on."}
        # ... (rest of the method is unchanged)
        # ...
        self.save_state()
        return {"status": "Training complete"}

    def get_graph_structure(self):
        return self.stag.get_flattened_structure()
