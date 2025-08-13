# Implements the hierarchical framework (STAG - Self-organizing Tree-like Adaptive Graph).
# This system organizes the GNGs into a multi-level taxonomy, inspired by GHSOM.
# It manages a tree of GNG instances, where high-error nodes in a parent GNG
# can expand into a new child GNG at the next level of the hierarchy.
# See Section 2.4 and 3.2 of the Project Chimera specification.

import numpy as np
from .gng_engine import GNG_Engine

class STAG_Framework:
    def __init__(self, dimensions, **kwargs):
        """
        Initializes the STAG Framework.

        Args:
            dimensions (int): The dimensionality of the input vectors.
            **kwargs: Hyperparameters for the GNGs and the STAG itself.
        """
        self.dimensions = dimensions
        self.gng_params = kwargs

        # The hierarchy is a tree structure where each node holds a GNG instance.
        self.tree = self._create_tree_node()

    def _create_tree_node(self, parent_gng_node_id=None):
        """Helper to create a new node in the hierarchy tree."""
        return {
            'gng': GNG_Engine(self.dimensions, **self.gng_params),
            'parent_node_id': parent_gng_node_id,
            'children': []  # List of child tree nodes
        }

    def find_terminal_node(self, input_vector):
        """
        Traverses the hierarchy to find the terminal GNG and winning node for an input.

        Args:
            input_vector (np.array): The input vector to route through the hierarchy.

        Returns:
            tuple: A tuple containing (
                terminal_level_node (dict): The tree node containing the terminal GNG.
                winner_id (int): The ID of the winning node within that GNG.
            )
        """
        current_level = self.tree
        while True:
            gng = current_level['gng']
            if len(gng.nodes) < 2:
                # This GNG is not yet populated enough to make a decision.
                return current_level, None

            winner_id, _ = gng._find_winners(input_vector)

            # Check if the winner node has a child GNG
            child_node = next((child for child in current_level['children'] if child['parent_node_id'] == winner_id), None)

            if child_node:
                # Traverse down the hierarchy
                current_level = child_node
            else:
                # This is the terminal node
                return current_level, winner_id

    def expand_node(self, parent_level_node, parent_gng_node_id):
        """
        Expands a node in a parent GNG into a new child GNG.
        This method creates the new child GNG and attaches it to the tree.
        The caller is responsible for training the new GNG.

        Args:
            parent_level_node (dict): The tree node from the hierarchy containing the parent GNG.
            parent_gng_node_id (int): The ID of the node within the parent GNG to expand.

        Returns:
            GNG_Engine: The new, untrained GNG instance for the child level.
        """
        print(f"Expanding node {parent_gng_node_id} in parent GNG.")

        # Check if a child for this node already exists
        if any(child['parent_node_id'] == parent_gng_node_id for child in parent_level_node['children']):
            # This should not happen if the agent logic is correct
            print(f"Warning: Expansion attempted on node {parent_gng_node_id} which already has a child.")
            return None

        new_level_node = self._create_tree_node(parent_gng_node_id=parent_gng_node_id)
        parent_level_node['children'].append(new_level_node)

        return new_level_node['gng']

    def get_serializable_structure(self):
        """
        Returns a serializable representation of the entire STAG tree.
        """
        def serialize_level(level_node):
            return {
                'gng_state': level_node['gng'].get_state(),
                'parent_node_id': level_node['parent_node_id'],
                'children': [serialize_level(child) for child in level_node['children']]
            }
        return serialize_level(self.tree)

    @classmethod
    def from_serializable_structure(cls, structure, **kwargs):
        """
        Creates a STAG instance from a serialized structure.
        """
        def deserialize_level(level_dict):
            gng = GNG_Engine.from_state(level_dict['gng_state'], **kwargs)
            node = {
                'gng': gng,
                'parent_node_id': level_dict['parent_node_id'],
                'children': [deserialize_level(child_dict) for child_dict in level_dict['children']]
            }
            return node

        stag = cls(structure['gng_state']['dimensions'], **kwargs)
        stag.tree = deserialize_level(structure)
        return stag
