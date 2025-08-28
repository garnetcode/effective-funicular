# Manages multiple STAG instances, one for each skill or environment.
# This allows the agent to learn and maintain distinct conceptual graphs for
# different tasks, preventing catastrophic forgetting.

import numpy as np
from .stag_framework import STAG_Framework
from .action.graph_planner import OptionModel

class SkillManager:
    def __init__(self, dimensions, **kwargs):
        """
        Initializes the Skill Manager.

        Args:
            dimensions (int): The dimensionality of the input vectors for the STAGs.
            **kwargs: Hyperparameters to be passed to each new STAG instance.
        """
        self.dimensions = dimensions
        self.gng_params = kwargs
        self.skill_graphs = {}  # A dictionary to hold STAG instances, keyed by skill_id
        self.option_models = {} # A dictionary to hold option models for each skill
        self.sf_learning_rate = kwargs.get('sf_learning_rate', 0.01)
        self.gamma = kwargs.get('gamma', 0.99)

    def _get_or_create_stag(self, skill_id):
        """
        Retrieves the STAG instance for a given skill_id.
        If it doesn't exist, a new one is created.
        """
        if skill_id not in self.skill_graphs:
            print(f"Creating new skill graph for skill_id: {skill_id}")
            self.skill_graphs[skill_id] = STAG_Framework(self.dimensions, **self.gng_params)
            self.option_models[skill_id] = {}
        return self.skill_graphs[skill_id]

    def find_terminal_node_and_path(self, skill_id, input_vector):
        """
        Delegates the call to the appropriate STAG instance for the given skill.
        """
        stag = self._get_or_create_stag(skill_id)
        return stag.find_terminal_node_and_path(input_vector)

    def expand_node(self, skill_id, parent_level_node, parent_gng_node_id):
        """
        Delegates the node expansion to the appropriate STAG instance.
        """
        stag = self._get_or_create_stag(skill_id)
        return stag.expand_node(parent_level_node, parent_gng_node_id)

    def prune_graph(self, skill_id, min_utility):
        """
        Delegates the pruning to the appropriate STAG instance.
        """
        stag = self._get_or_create_stag(skill_id)
        stag.prune_graph(min_utility)

    def get_flattened_structure(self, skill_id):
        """
        Gets the flattened graph structure for a specific skill.
        """
        stag = self._get_or_create_stag(skill_id)
        return stag.get_flattened_structure()

    def find_k_nearest_neighbors(self, skill_id, vector, k=5):
        """Delegates the k-NN search to the appropriate STAG instance."""
        stag = self._get_or_create_stag(skill_id)
        return stag.find_k_nearest_in_terminal_gng(vector, k)

    def update_option_model(self, skill_id, from_node, to_node, reward, duration):
        """Updates the option model for a transition between two nodes."""
        if skill_id not in self.option_models:
            self.option_models[skill_id] = {}

        option_key = (from_node, to_node)
        if option_key not in self.option_models[skill_id]:
            self.option_models[skill_id][option_key] = OptionModel(from_node, to_node)

        self.option_models[skill_id][option_key].update(reward, duration)

    def update_successor_features(self, skill_id, from_node_id, to_node_id, state_features):
        """Updates the successor features for a transition between two nodes."""
        stag = self._get_or_create_stag(skill_id)
        # This assumes the nodes are in the terminal GNG of the STAG
        terminal_gng = stag.level_map[max(stag.level_map.keys())]

        if from_node_id in terminal_gng.nodes and to_node_id in terminal_gng.nodes:
            from_node = terminal_gng.nodes[from_node_id]
            to_node = terminal_gng.nodes[to_node_id]

            # TD update for successor features
            target_psi = state_features + self.gamma * to_node['psi']
            from_node['psi'] += self.sf_learning_rate * (target_psi - from_node['psi'])

    def get_serializable_structure(self):
        """
        Returns a serializable representation of all skill graphs and their associated option models.
        """
        serializable_option_models = {}
        for skill_id, options in self.option_models.items():
            serializable_option_models[skill_id] = {}
            for (from_node, to_node), model in options.items():
                # Convert tuple key to a string for JSON compatibility
                str_key = f"{from_node},{to_node}"
                serializable_option_models[skill_id][str_key] = model.to_dict()

        return {
            'dimensions': self.dimensions,
            'skill_graphs': {sid: stag.get_serializable_structure() for sid, stag in self.skill_graphs.items()},
            'option_models': serializable_option_models,
        }

    @classmethod
    def from_serializable_structure(cls, structure, **kwargs):
        """
        Creates a SkillManager instance from a serialized structure.
        """
        # Prioritize the dimension passed directly from the agent constructor,
        # fallback to the one saved in the structure. This handles legacy save files.
        dimensions = kwargs.get('dimensions', structure.get('dimensions'))
        if dimensions is None:
            raise ValueError("SkillManager deserialization failed: 'dimensions' is None.")

        # Avoid passing 'dimensions' as a keyword argument if it's already being passed positionally.
        kwargs.pop('dimensions', None)
        manager = cls(dimensions, **kwargs)

        # Ensure the dimensions are passed down to the STAG frameworks.
        # This is critical for deserializing older save files where individual
        # STAG structures might not have the 'dimensions' key.
        downstream_kwargs = kwargs.copy()
        downstream_kwargs['dimensions'] = dimensions

        # Load skill graphs
        sf_dimension = kwargs.get('sf_dimension', 64) # Default from ChimeraAgent
        for skill_id, stag_data in structure.get('skill_graphs', {}).items():
            stag = STAG_Framework.from_serializable_structure(stag_data, **downstream_kwargs)
            # AGENT_FIX: Add backward compatibility for 'psi' key in GNG nodes
            if max(stag.level_map.keys()) in stag.level_map:
                terminal_gng = stag.level_map[max(stag.level_map.keys())]
                for node_id, node_data in terminal_gng['gng'].nodes.items():
                    if 'psi' not in node_data:
                        node_data['psi'] = np.zeros(sf_dimension)
            manager.skill_graphs[skill_id] = stag


        # Load option models
        for skill_id, options_data in structure.get('option_models', {}).items():
            manager.option_models[skill_id] = {}
            for str_key, model_data in options_data.items():
                from_node, to_node = map(int, str_key.split(','))
                manager.option_models[skill_id][(from_node, to_node)] = OptionModel.from_dict(model_data)

        return manager
