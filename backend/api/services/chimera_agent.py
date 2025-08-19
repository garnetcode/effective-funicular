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
            self.action_head = ActionHead(
                input_dim=dimensions,
                n_actions=n_actions,
                learning_rate=self.learning_rate
            )

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
            'action_head_state': self.action_head.get_state(),
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
            self.action_head = ActionHead(
                self.dimensions,
                self.n_actions,
                learning_rate=self.learning_rate
            )
            # The action_head_state is a raw object from np.load, not a dict
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

    def is_novel(self, embedding, novelty_threshold=0.2):
        """
        Checks if an embedding is novel compared to existing patterns.
        A simple approach: check the distance to the nearest pattern.
        """
        if not self.patterns:
            return True

        all_patterns = np.array(list(self.patterns.values()))
        distances = np.linalg.norm(all_patterns - embedding, axis=1)
        min_distance = np.min(distances)

        return min_distance > novelty_threshold

    def organize_memory(self, pattern_id, reward=0):
        """
        ORGANIZE Mode (Sec 3.3): Updates the GNG/STAG structure.
        The reward parameter influences the utility of the winning neuron.
        """
        if pattern_id not in self.patterns:
            raise ValueError(f"Pattern ID {pattern_id} not found.")

        cue_vector = self.patterns[pattern_id]

        # 1. Get stable attractor from Hopfield network
        stable_attractor = self.hopfield.recall(cue_vector)

        # 2. Find the terminal GNG and winner node for this attractor
        terminal_node, winner_id, _ = self.stag.find_terminal_node_and_path(stable_attractor)
        if winner_id is None: # GNG not initialized enough
            terminal_node['gng'].process_input(stable_attractor)
            self.save_state()
            return {"status": "Organization step skipped, GNG too small."}

        # 3. Update the mapping for this pattern with a robust identifier
        level_id = terminal_node['level_id']
        self.pattern_node_map[pattern_id] = (level_id, winner_id)

        # 4. Process the input in the terminal GNG, modulated by reward
        terminal_gng = terminal_node['gng']
        terminal_gng.process_input(stable_attractor, reward=reward)

        # 5. Check for expansion condition
        if terminal_gng.nodes[winner_id]['error'] > self.stag_expansion_threshold:
            self._trigger_expansion(terminal_node, winner_id)

        self.save_state()
        return {"status": "Organization step complete."}

    def _trigger_expansion(self, parent_level_node, parent_gng_node_id):
        """
        Orchestrates the expansion of a GNG node into a new child GNG.
        """
        parent_level_id = parent_level_node['level_id']

        # 1. Find all patterns that belong to the node being expanded
        patterns_to_remap = []
        for pid, mapped_id in self.pattern_node_map.items():
            # Check if the mapped_id is for the node being expanded
            if isinstance(mapped_id, tuple) and mapped_id == (parent_level_id, parent_gng_node_id):
                patterns_to_remap.append(pid)

        if not patterns_to_remap:
            print(f"Node {parent_gng_node_id} in level {parent_level_id} triggered expansion but no patterns mapped to it. Skipping.")
            return

        # 2. Create the new child GNG in the STAG framework
        new_child_gng = self.stag.expand_node(parent_level_node, parent_gng_node_id)
        if new_child_gng is None: return

        new_level_id = self.stag._next_level_id - 1 # The ID of the GNG we just created

        # 3. Train the new child GNG with the identified patterns and remap them
        print(f"Training new child GNG for level {new_level_id} with {len(patterns_to_remap)} patterns.")
        for pattern_id in patterns_to_remap:
            pattern_vector = self.patterns[pattern_id]
            stable_attractor = self.hopfield.recall(pattern_vector)

            # Train the new GNG
            new_child_gng.process_input(stable_attractor)

            # Re-map the pattern to its new home in the child GNG
            new_winner_id, _ = new_child_gng._find_winners(stable_attractor)
            self.pattern_node_map[pattern_id] = (new_level_id, new_winner_id)
            print(f"Re-mapped pattern {pattern_id} to new node ({new_level_id}, {new_winner_id})")

    def update_cortex_config(self, new_cortex_configs):
        """Merges new cortex configs, re-initializes cortexes, and saves the agent."""
        self.cortex_configs.update(new_cortex_configs)
        self.cortexes = _initialize_cortexes(self.cortex_configs, self.dimensions)
        self.save_state()
        print(f"Agent {self.agent_id} cortex config updated to: {self.cortex_configs}")

    def update_action_space(self, n_actions):
        """Resets the agent's action head for a new number of actions."""
        self.n_actions = n_actions
        self.action_head = ActionHead(
            input_dim=self.dimensions,
            n_actions=self.n_actions,
            learning_rate=self.learning_rate
        )
        self.save_state()
        print(f"Agent {self.agent_id} action space updated to: {n_actions} actions")

    def perceive(self, cortex_id, raw_input):
        if cortex_id not in self.cortexes: raise ValueError(f"Cortex '{cortex_id}' not found.")
        return self.cortexes[cortex_id].process(raw_input)

    def get_internal_state_representation(self, input_embedding):
        stable_attractor = self.hopfield.recall(input_embedding)
        terminal_node, winner_id, _ = self.stag.find_terminal_node_and_path(stable_attractor)
        if winner_id is not None and winner_id in terminal_node['gng'].nodes:
            return terminal_node['gng'].nodes[winner_id]['weight']
        return stable_attractor

    def get_state_novelty_error(self, input_embedding):
        """
        Calculates the novelty of a state based on the GNG error of the
        winning node. A high error indicates a novel, poorly understood state.
        """
        stable_attractor = self.hopfield.recall(input_embedding)
        terminal_node, winner_id, _ = self.stag.find_terminal_node_and_path(stable_attractor)
        if winner_id is not None and winner_id in terminal_node['gng'].nodes:
            return terminal_node['gng'].nodes[winner_id]['error']
        # Return a high error if no node is found, encouraging exploration
        return 1.0

    def probe_activity(self, cortex_id, raw_input):
        """
        Processes an input and returns the activation path through the STAG hierarchy.
        """
        # 1. Perceive the input to get an embedding
        embedding = self.perceive(cortex_id, raw_input)

        # 2. Get the stable attractor state from the Hopfield network
        stable_attractor = self.hopfield.recall(embedding)

        # 3. Find the activation path in the STAG framework
        _, _, activation_path = self.stag.find_terminal_node_and_path(stable_attractor)

        return {"activation_path": activation_path}

    def select_action(self, state_embedding):
        internal_state = self.get_internal_state_representation(state_embedding)

        # Convert to tensor for PyTorch model
        state_tensor = torch.from_numpy(internal_state).float().unsqueeze(0)

        with torch.no_grad():
            action_logits = self.action_head(state_tensor).squeeze(0)

        # Convert back to numpy for softmax and sampling
        action_probs_np = softmax(action_logits.numpy())
        action = np.random.choice(self.n_actions, p=action_probs_np)

        # We need the log_prob as a tensor for the loss calculation
        log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
        action_log_prob = log_probs[action]

        return action, action_log_prob, internal_state

    def record_experience(self, internal_state, action, log_prob, reward, pattern_id=None):
        self.episode_memory.append({
            "state": internal_state,
            "action": action,
            "log_prob": log_prob,
            "reward": reward,
            "pattern_id": pattern_id
        })

    def train(self):
        # This is policy/RL training, separate from unsupervised cognitive learning
        if not self.episode_memory:
            return {"status": "No experiences to train on."}

        # REINFORCE algorithm implementation with manual gradient calculation
        G = 0
        returns = []
        # Calculate discounted returns (rewards-to-go)
        for step in reversed(self.episode_memory):
            G = step['reward'] + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns)
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-9)

        # Process memory organization from the episode
        for i, step in enumerate(self.episode_memory):
            pattern_id = step.get('pattern_id')
            if pattern_id is not None:
                self.organize_memory(pattern_id, reward=returns[i])

        # Policy gradient update
        log_probs = torch.stack([step['log_prob'] for step in self.episode_memory])
        returns_tensor = torch.from_numpy(returns).float()

        policy_loss = (-log_probs * returns_tensor).mean()

        self.action_head.optimizer.zero_grad()
        policy_loss.backward()
        self.action_head.optimizer.step()

        # Clear memory for the next episode
        self.episode_memory = []

        self.save_state()
        return {"status": "Training complete", "loss": policy_loss.item()}

    def consolidate_memories(self, n_replays=1):
        """
        Performs offline memory consolidation by replaying existing patterns.
        This strengthens the representations in the STAG/GNG framework.
        """
        if not self.patterns:
            return {"status": "No patterns to consolidate."}

        print(f"Starting memory consolidation for {len(self.patterns)} patterns...")
        all_pattern_ids = list(self.patterns.keys())

        for i in range(n_replays):
            # Shuffle the patterns for each replay epoch
            np.random.shuffle(all_pattern_ids)
            for pattern_id in all_pattern_ids:
                # We add a small amount of noise to the cue to promote robustness
                # without corrupting the core memory.
                original_pattern = self.patterns[pattern_id]
                noise = np.random.normal(0, 0.01, self.dimensions)
                noisy_cue = original_pattern + noise

                # The core of consolidation is re-organizing the memory, which
                # strengthens the appropriate nodes in the GNG.
                # We use the noisy cue for recall but the original pattern ID
                # for the organization process.
                self.organize_memory(pattern_id)

        self.save_state()
        print("Memory consolidation complete.")
        return {"status": "Consolidation complete."}

    def get_graph_structure(self):
        return self.stag.get_flattened_structure()
