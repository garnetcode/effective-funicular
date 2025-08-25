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

    def update_option_model(self, skill_id, from_node, to_node, reward, duration):
        """Updates the option model for a transition between two nodes."""
        if skill_id not in self.option_models:
            self.option_models[skill_id] = {}

        option_key = (from_node, to_node)
        if option_key not in self.option_models[skill_id]:
            self.option_models[skill_id][option_key] = OptionModel(from_node, to_node)

        self.option_models[skill_id][option_key].update(reward, duration)

    def get_serializable_structure(self):
        """
        Returns a serializable representation of all skill graphs.
        NOTE: Option models are not currently serialized.
        """
        all_skill_data = {}
        for skill_id, stag_instance in self.skill_graphs.items():
            all_skill_data[skill_id] = stag_instance.get_serializable_structure()
        return {
            'dimensions': self.dimensions,
            'skill_graphs': all_skill_data
        }

    @classmethod
    def from_serializable_structure(cls, structure, **kwargs):
        """
        Creates a SkillManager instance from a serialized structure.
        """
        dimensions = structure.get('dimensions')
        manager = cls(dimensions, **kwargs)

        serializable_graphs = structure.get('skill_graphs', {})
        for skill_id, stag_data in serializable_graphs.items():
            manager.skill_graphs[skill_id] = STAG_Framework.from_serializable_structure(stag_data, **kwargs)
            if skill_id not in manager.option_models:
                manager.option_models[skill_id] = {}


        return manager
